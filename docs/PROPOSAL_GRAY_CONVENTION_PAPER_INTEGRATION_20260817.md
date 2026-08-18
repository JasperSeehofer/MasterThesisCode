# PROPOSAL — Gray-convention finding: paper integration (row #110 item 4 paper-thread task)

**Date:** 2026-08-17 · **Status: PROPOSAL, PRESENTED, NOT ADJUDICATED.** Propose-only artifact:
no ledger row, no paper source, and no existing file is touched by this document. The decision
table is §5; the author decides, this document presents.

**Mandate chain (verbatim quotes per the attribution convention):**

- **Row #110 item 4** (author's verbatim ruling 2026-08-15: *"all approved"*; itemisation
  orchestrator-derived, ledger text): "**[RULE — granted, branch reading orchestrator-derived]**
  The Gray-convention finding **enters the paper's scope**, with the FULL-B/D/F chain as its
  quantitative backbone (what the published convention costs at σ_z > 0). *Reading flagged for
  author veto; concrete paper integration is a paper-thread task.*" — This document IS that
  paper-thread task.
- **Row #117 item 2** (author's verbatim ruling 2026-08-17: *"please note all as ratified"*;
  itemisation is the fusion proposal's own table): "item 2 [P3] catalogue-leg fork deferred to
  the Gray-convention paper task unless the counterfactual shows material mixture skew".
- **Row #119 item 1** (author's verbatim ruling 2026-08-17: *"as recommended please"*;
  interpretation basis orchestrator-derived, ledger text): "**[RULE] M-4 materiality — NOT
  MATERIAL.** The measured mixture skew (median +0.02–0.03, max +0.204 catalogue-share gain,
  confined to the 161/159 of 1588 catalogue-bearing events) does not trigger row #117 item 2's
  'unless material' condition. **The [P3] catalogue-leg fork stays deferred to the
  Gray-convention paper task (row #110)**, which now holds the M-4 numbers as its quantitative
  input."

All ledger quotes: `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`.

---

## 1. The Gray-convention finding — self-contained explainer

### 1.1 What the published convention is

Gray et al. 2020 (arXiv:1908.06050, PRD 101, 122001), Eq. (A.10), together with
Mandel–Farr–Gair 2019, prescribe the **denominator-only selection convention** for dark-siren
likelihoods: the detection probability p_det enters the per-event likelihood *only* through the
selection normalization (the denominator / α-term); putting a p_det factor in the numerator is
called out as "the most common mistake", biasing H0 high. Our production code implements this
convention verbatim, with an explicit comment at the catalogue leg
(`darksiren_emri/bayesian_inference/bayesian_statistics.py`, comment at `:5204` at the L6-DER3
commit, **`:5328` at current HEAD** post-`2b10b8b8` — verified: *'"most common mistake"
(arXiv:1809.02063) and biases H0 high'*; quoted in full via
`results/mechanism_study_20260813/L6_DER3_PRODUCTION_COMPLETION_LEG_20260816.md` §2,
commit `e3eec5c0`).
The paper skeleton already cites and follows this convention throughout
(`paper/sections/method.tex:283`, `discussion.tex:42–61`, `results.tex:257–258`).

### 1.2 What the finding is: the convention is detection-model-conditional

The load-bearing structural result (L6-DER3 §3, `L6_DER3_PRODUCTION_COMPLETION_LEG_20260816.md`):

- Under **data-deterministic detection** (detection a deterministic function of the observed
  data, e.g. a threshold on observed SNR), `p(data|θ,det) = p(data|θ)/P(det|θ)` on the detected
  support, the numerator p_det cancels, and the MFG/Gray denominator-only form is **exact**. The
  "most common mistake" warning is correct in this model.
- Under **latent-thresholded detection** (detection random given the inference coordinates
  (z, M), independent of the measurement noise — which is what our pipeline actually implements:
  the SNR threshold acts on the full parameter vector θ, and `SimulationDetectionProbability` is
  the survival `P(SNR(θ) ≥ 20 | d_L, M_z)` with randomness from the marginalized extrinsic
  parameters), the correct per-event likelihood is the **selected-prior form**: p_det stays
  inside the numerator's population integral, paired against its own normalization α(h):

      L_e ∝ (1/K) Σ_k (1/imp_k) ∫ dz [w_pop(z;h)·S̄_φ(z;h)/α(h)] · N(z_obs,k; z, σ_k)
                                        · N(d_obs; d_L(z,h), σ_d·d_L(z,h))

  (the FULL-F form, `results/mechanism_study_20260813/DRAFT_A_FULL_ESTIMATOR_20260815.md` §1 +
  addendum A1, commits `fe172d6f` + `860b9d3f`). The denominator-only convention, applied in a
  latent-thresholded pipeline, keeps the −N ln α term while omitting the paired numerator weight
  w_pop·S̄_φ — a **broken pairing**, not two independent defects (draft §1 ingredient 2:
  "D1+D4 are one defect, not two").

The finding, in one sentence: **Gray's denominator-only convention is exact only under
data-deterministic detection; under the latent-thresholded detection model that our pipeline
(and, structurally, any pipeline whose detection is thresholded on latent parameters with
marginalized extrinsics) actually realizes, it drops the selection-weighted population prior
from the numerator, and at σ_z > 0 that omission carries a measurable H0 bias.**

On the σ_z > 0 conditionality (orchestrator-derived reading of row #110's wording, flagged):
the omitted weight acts through the numerator's z-marginalization over the photo-z kernel; with
σ_z → 0 the kernel collapses and the weight tends to a per-event constant that normalizes out.
The venue measurements below are all at the production-like photo-z scatter. UNVERIFIED as a
measured limit: no σ_z → 0 cell was run; a spec-z-kernel venue cell would verify it.

### 1.3 What it costs — the FULL-B/D/F chain (the quantitative backbone)

Venue: mirror geometry, 15 MN0X seed replays, 1D channel, full dose. Source:
`DRAFT_A_FULL_ESTIMATOR_20260815.md` §2 + addendum (commits `fe172d6f`, `860b9d3f`).

| candidate | ingredients | paired venue tilt T (nats/h), f_i = 1.0 |
|---|---|---:|
| **coded base** (denominator-only convention, as published) | ratio-pdf GW, bare kernel, −N ln α unpaired | **+2644.0 ± 46.5** |
| FULL-A | d_obs-density GW factor only | +2529.4 |
| FULL-B | + w_pop numerator weight (pairing restored, population only) | **−103.6 ± 46.2** |
| FULL-D | + S̄_φ (full selected prior w_pop·S̄_φ/α) | **+183.4 ± 47.0** |
| **FULL-F** (completed candidate: D × leave-one-out 1/imp_k) | | **+30.6 ± 42.7 — zero-consistent (0.7σ)** |

- The coded-base tilt +2644 nats/h corresponds to the measured venue H0 MAP bias **+0.0373**
  (displacement law Ā ≈ 7.0×10⁴, draft §2 item 2); FULL-F's implied bias ≈ **+0.0004** — an
  **~86× reduction**, and the pairing alone (FULL-B/D) is a 14–25× collapse.
- FULL-A isolates the point: the density-form GW repair alone changes almost nothing
  (+2529 vs +2644) — **the cost is owned by the missing population pairing**, i.e. by the
  convention, not by a numerical detail.
- Kernel renormalization was staged and **refuted** (FULL-C/E overshoot by ≈ −1100 nats/h,
  draft §2 item 3) — the finding is not "add more factors"; it is precisely the α-pairing.
- Registered confirmation on fresh seeds (A-FULL arm, ledger row #111, commit `715943ca`,
  `results/mechanism_study_20260813/STAGE5_READOUT.md`): T(1D) = **+22.0 ± 29.2** (DS-F1 PASS),
  bias **+0.0010 ± 0.0011** (from +0.0373), **1D coverage RESTORED**
  (0.64/0.76/0.96 vs nominal 0.50/0.68/0.90; every prior arm 0/25).
- 2D channel (mass channel): the same fork, fused form `g_sel` (single ∫dM φ·p_det·N instead of
  the factorized S̄_φ×g). Venue arm A-FULL-2D (ledger row #116, commit `bcd66529`,
  `AFULL2D_ARM_READOUT_20260817.md`): DS-G1 PASS **−11.8 ± 0.61 nats/h** in the registered band,
  2D bias **+0.0006 ± 0.0013**, coverage restored, 1D bit-identical.
- Literature anchor (`L0_LIT_FULLTEXT_20260815.md`, full-text read of Gray 2020 + Gray 2023
  arXiv:2308.02281): neither paper's event term carries the numerator selection weight; Gray
  2023's renormalization escape clause is conditional on a shared-object symmetry our
  per-candidate construction does not satisfy. No published quantification of this bias class
  was found — the finding is, to the extent of that read, **novel**.

### 1.4 What production paid (the fused estimator is now the default)

The [P1]+[P2] completion-leg fusion shipped as `[PHYSICS]` commit `2b10b8b8` (rows #117–#118).
The production counterfactual (rows #119; `results/run_20260817_fusion_counterfactual/CAMPAIGN_REPORT_20260817.md`,
readout commit `7b512877`; prereg `a6a98d2a` + `ac24b632`; 2 cells × 2 venues × 41 h-points ×
1588 events):

- **M-2, 1D channel:** Σ Δln tilt fused−off **+24.588 / +22.736** chord nats/h (iiib / joint_r1),
  **+30.9 / +32.3** central at h = 0.73 — bit-consistent with the independent N-2 measurement
  (`results/run_20260805_n2sel1d/readout.json`, claim status DRAFT).
- **M-1, 2D channel:** **+1.245 / −3.268** chord nats/h — near-inert, exactly as the
  sharp-likelihood regime call (row #118 MAJOR-1) predicted; the venue's −11.8 does NOT transfer
  (A3 discipline: venue magnitudes never carry).
- **M-3:** **zero MAP motion** in every channel × venue; 1D width tightens
  (σ 0.0068→0.0053 iiib, 0.0086→0.0065 joint_r1), 2D width unchanged. The 1D MAP stays railed
  at 0.600 — the rail is owned by photo-z (ledger #36), not by the convention.
- Shape: *suppression-without-motion* — S-factors < 1 drop every likelihood LEVEL (median Δln
  at 0.73: −1.32 (1D), −0.41 (2D)); only the h-slope matters for H0, and it is 1D-carried.

**Paper-facing summary of §1:** the convention costs +0.037 in H0 in the venue where the full
chain isolates it; in the production campaign at production dose the paired repair moves tilts
by ~+25 nats/h (1D) / ±3 (2D) and no posterior of record moves — the published campaign numbers
survive the fix, and the bridge is the counterfactual, not a re-run (row #119 item 2).

---

## 2. The M-4 attachment and the [P3] catalogue-leg fork

### 2.1 [P3] — precise definition

From `docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md` (commit `298c4963`) §"[P3]
Catalogue leg: the per-host selection weighting (the Gray-convention fork)", as amended by its
verifier addendum (commit `44aa239e`, MAJOR-3):

> under the latent model each catalogued candidate's numerator gains p_det weighting against
> its own mass marginal (`mz_integral` → single-∫dM with `S_4D`; 1D leg a per-host S̄-type
> factor) — contradicting the coded `:5204` convention ("a numerator p_det is the MFG most
> common mistake") which presumes data-deterministic detection.

I.e. [P3] is the **same fork as §1.2, applied per-galaxy to the catalogue leg** — the one
numerator the fusion (`2b10b8b8`) deliberately left in the published Gray/MFG convention.
Mixture consistency is the forcing function: with [P2] on, the S̄-free catalogue leg is
**OVER-weighted** relative to the fused completion leg wherever S̄_φ < 1 (sign per MAJOR-3;
the proposal body's "DOWN-weighted" is inverted and superseded). Rows #117 item 2 / #118 G3
deferred it here unless the counterfactual showed material skew.

### 2.2 M-4 — the measured size of the skew

`CAMPAIGN_REPORT_20260817.md` §6 scorecard + §7 vocabulary (commit `7b512877`), at h = 0.73:

| quantity | iiib | joint_r1 |
|---|---|---|
| mean Δshare_cat (fused−off) | +6.1e-3 | +5.7e-3 |
| movers (events with any Δ) | 161 / 1588 | 159 / 1588 |
| median Δshare_cat among movers | +0.034 | +0.022 |
| max Δshare_cat | +0.204 | +0.203 |

where share_cat = A_cat/(A_cat+B_num), the catalogue fraction of the 1D mixture; median
share_cat is 0 (most events completion-dominated), and the ~160 movers have median share 0.62.
Row #119 item 1 ruled this **NOT MATERIAL** for production code — and assigned the numbers to
this task as its quantitative input.

### 2.3 What M-4 adds to the paper story

- It **closes the quantitative loop honestly**: the paper can state that one leg of the
  published convention is retained (catalogue numerator), and give the *measured* cost of
  retaining it — a ≤ +0.20 catalogue-share re-weighting confined to the ~10% of events that
  have a catalogue leg at all, with zero posterior motion (M-3).
- It supplies the **materiality argument** for why the retained convention does not contaminate
  the headline H0 numbers: the mixture skew never reaches the posterior of record.
- It sharpens the finding's scope line: the convention's measurable cost lives in the
  **completion (out-of-catalogue) population term**, which dominates at EMRI distances against
  GLADE+; the per-galaxy catalogue term is where the convention is cheapest in this regime —
  a genuinely useful message for ground-based analyses (catalogue-dominated regimes may differ;
  UNVERIFIED here — stating it as a caveat, not a claim, unless a dedicated cell is run).

### 2.4 How [P3] can appear in the paper — the fork's presentation options (author's choice, §5 item 2)

- **(a) Documented-convention option:** present [P3] as an identified, measured, deliberately
  retained convention choice — the paper keeps the Gray/MFG catalogue numerator, cites M-4 as
  the measured immateriality bound, and lists the per-galaxy fused form as future work. No new
  code, no new run. (Consistent with row #119 item 1's "stays deferred"; deferral of the *code*
  fork does not by itself dictate the *presentation*, hence this fork returns as a fresh [RULE].)
- **(b) Resolved-in-paper option:** the paper presents the latent-model derivation as the
  correct form for BOTH legs, and the retained catalogue convention as an approximation with
  the M-4 bound. Stronger claim; requires the derivation section to carry the per-galaxy form
  explicitly (no run needed, but the [P4]/V2 measure-prefactor question — G2's tracked
  systematic, material only in the catalogue leg's broad σ_M — must then be discussed in text).
- **(c) Silent option:** the paper describes only the shipped estimator and omits the fork.
  Cheapest; forfeits the novelty claim of §1.3 and sits in tension with row #110 item 4's
  scope grant. Listed for completeness.

---

## 3. Proposed concrete paper integration

The paper exists: `paper/main.tex` + `paper/sections/{introduction,method,results,discussion,conclusions,appendix_parameters}.tex`
(REVTeX-style skeleton; `book/` is the design-doc/site build, not the paper). Gray et al. is
already cited at `method.tex:164–165,283`, `results.tex:257`, `discussion.tex:42–61,191,235`,
`conclusions.tex:21`, `introduction.tex:93–94`. Note `discussion.tex:235` currently promises
"the bias reduction from the Gray et al. correction" from a production run — the finding gives
that sentence a precise, measured content and partially supersedes its framing.

Proposed placement (bullet-level skeleton, not final prose; all numbers with §1/§2 provenance):

1. **Method, `sec:likelihood` (after `eq:completeness_combination`/`eq:Lcomp`,
   `method.tex:283–313`) — new subsubsection "Selection convention and the detection model":**
   - state the two detection models (data-deterministic vs latent-thresholded) and which one
     the pipeline realizes; give the selected-prior likelihood (§1.2 display) as the estimator
     actually used for the completion legs (it IS production since `2b10b8b8`);
   - one paragraph on why the denominator-only form is exact in one model and not the other
     (the broken α-pairing), citing Gray 2020 Eq. (A.10) and MFG 2019;
   - state the catalogue-leg convention per the §5 item 2 ruling.
2. **Results, `sec:systematics` (`results.tex:301`) — new named systematic "Selection-convention
   bias at σ_z > 0":**
   - the FULL-B/D/F ladder table (§1.3, condensed: coded base +2644 ↔ bias +0.0373 → FULL-F
     +30.6 ± 42.7 ↔ +0.0004, ~86×; registered-arm confirmation +0.0010 ± 0.0011, coverage
     restored; 2D: −11.8 ± 0.61 in band, bias +0.0006 ± 0.0013);
   - the production counterfactual paragraph (§1.4: 1D +24.6/+22.7 chord, 2D near-inert, zero
     MAP motion — the campaign posteriors stand; counterfactual as the recorded bridge);
   - the M-4 paragraph (§2.2 numbers + §2.3 materiality argument), phrased per the item 2 ruling.
3. **Discussion, `sec:comparison` (`discussion.tex:12`) — extend the existing Gray comparison
   paragraph (`discussion.tex:42–61`):** the convention is inherited from ground-based practice;
   its validity is detection-model-conditional; at EMRI photo-z scatter the completion-term cost
   is the dominant, measured one; no published quantification found (L0-LIT provenance).
4. **Discussion, `sec:caveats`:** the σ_z → 0 limit unmeasured; catalogue-dominated regimes
   unprobed; pool-vs-model mismatch and −11.7-class residual carried; #66/#67 production
   calibration (pp_coverage mass channel) TO-BUILD — magnitude ≠ calibration
   (report §9 flag 4).
5. **Conclusions (`conclusions.tex:21` vicinity):** one sentence per the §5 item 1 scope ruling.

**Figures/tables:**

| artifact | status | source |
|---|---|---|
| FULL-B/D/F ladder table (tilt + implied bias per candidate) | TO-MAKE (trivial: table, numbers of record) | `DRAFT_A_FULL_ESTIMATOR_20260815.md` §2 + addendum |
| Tilt-collapse ladder figure (coded base → A → B/D → F, with the registered-arm point) | TO-MAKE | same + `STAGE5_READOUT.md` |
| Dose-surface figure | EXISTS, reusable if the dose structure is presented | `results/mechanism_study_20260813/fig_dose_surface_20260814.pdf` (+ `.json`) |
| M-4 Δshare_cat distribution (movers histogram, both venues) | TO-MAKE | `results/run_20260817_fusion_counterfactual/readout.json` |
| Counterfactual tilt/posterior overlay (off vs fused per channel × venue) | TO-MAKE (optional; the M-1..M-3 table may suffice) | same `readout.json` |
| Existing paper figures | untouched | `paper/figures/*.pdf` |

No new computation is required for any TO-MAKE item — all are renderings of committed JSON/MD
numbers. Any *new measurement* (σ_z → 0 cell, catalogue-dominated cell) is out of scope here
and would return as a fresh [DO].

---

## 4. Provenance ledger for every number used above

| number | file | commit |
|---|---|---|
| +2644.0 ± 46.5 / +2529.4 / −103.6 ± 46.2 / +183.4 ± 47.0 / FULL-C/E ≈ −1100 swing | `results/mechanism_study_20260813/DRAFT_A_FULL_ESTIMATOR_20260815.md` §2 | `fe172d6f` |
| FULL-F +30.6 ± 42.7 (f_i=1.0), +168.9 ± 58.8 (f_i=0.25); ~86×; bias +0.0004; KS D=0.085 | same, addendum | `860b9d3f` |
| bias +0.0373 ↔ T +2625/+2644; Ā ≈ 7.0×10⁴ | same §2 item 2 | `fe172d6f` |
| A-FULL arm: +22.0 ± 29.2; +0.0010 ± 0.0011; coverage 0.64/0.76/0.96 | `STAGE5_READOUT.md` §3 / ledger row #111 | `715943ca` |
| A-FULL-2D: −11.8 ± 0.61; +0.0006 ± 0.0013; coverage restored | `AFULL2D_ARM_READOUT_20260817.md` / ledger row #116 | `bcd66529` |
| fusion shipped | production `[PHYSICS]` commit | `2b10b8b8` |
| M-1 +1.245/−3.268; M-2 +24.588/+22.736 (chord), +30.9/+32.3 (central); M-3 zero MAP, widths 0.0068→0.0053 / 0.0086→0.0065; medians Δln −1.32/−0.41 | `results/run_20260817_fusion_counterfactual/CAMPAIGN_REPORT_20260817.md` §4/§6 | `7b512877` |
| M-4: +6.1e-3/+5.7e-3 mean; 161/159 of 1588; med +0.034/+0.022; max +0.204/+0.203; movers' median share 0.62 | same §6/§7 | `7b512877` |
| [P3] definition + MAJOR-3 sign flip; G2 V2 bound ≲1e-6 at σ_cond p50 8.8e-8 | `docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md` + addendum / ledger row #118 | `298c4963`, `44aa239e` |
| latent-vs-data-deterministic fork; `:5204` comment (now `:5328`); Gray Eq. (A.10) | `L6_DER3_PRODUCTION_COMPLETION_LEG_20260816.md` §2–§3 | `e3eec5c0` |
| literature novelty read | `L0_LIT_FULLTEXT_20260815.md` | `9bf7938b` |

**UNVERIFIED items:** (i) the σ_z → 0 no-cost limit (orchestrator-derived reading of row
#110's "at σ_z > 0"; verifiable with a spec-z-kernel venue cell); (ii) the
catalogue-dominated-regime extrapolation (§2.3, flagged as caveat-only); (iii) the exact Gray
2020 Eq. (A.10) equation text (read via `pdftotext` in L0-LIT, `9bf7938b` — re-check against
the published PRD version before the paper quotes it).

---

## 5. DECISION TABLE (inline, per CLAUDE.md "Proposing decisions"; itemisation orchestrator-derived)

| # | decision | options / scope | tag |
|---|---|---|---|
| 1 | **Scope approval:** integrate the Gray-convention finding into the paper per §3's placement (method convention subsubsection + systematics entry + discussion comparison/caveats + conclusions sentence), with the §1 numbers as the quantitative backbone | approve as proposed / approve with edits / narrower scope (e.g. systematics + discussion only) | **[DO]** |
| 2 | **[P3] presentation choice** (§2.4) — how the retained catalogue-leg convention appears in the paper. The code-fork deferral is already ruled (row #119 item 1); THIS is the presentation ruling on inputs (M-4) that did not exist when row #110 was granted, so it returns fresh per the binding default. Options: (a) documented-convention (retained + M-4 bound, fused per-galaxy form as future work), (b) resolved-in-paper (latent-model form presented as correct for both legs; retained leg as measured approximation; V2 prefactor discussed), (c) silent | author picks (a)/(b)/(c) — no recommendation is binding; note (c) sits in tension with row #110 item 4's scope grant | **[RULE]** |
| 3 | **σ_z > 0 conditionality wording:** may the paper state the σ_z → 0 no-cost limit as a structural argument (unmeasured, caveated), or must it be measured first (spec-z-kernel venue cell — a new run, fresh [DO] if chosen) | caveated-statement / measure-first / omit | **[RULE]** |
| 4 | **TO-MAKE figure work** (§3 table): ladder table + tilt-collapse figure + M-4 histogram (all renderings of committed numbers, no new computation); optional counterfactual overlay | approve all / subset / tables-only | **[DO]** |
| 5 | **`discussion.tex:235` supersession:** rewrite the "production run underway … will quantify the bias reduction" future-work paragraph to reflect that the quantification now exists (§1.3–§1.4) | approve rewrite / keep and add | **[DO]** |

Nothing here authorizes production code changes, new cluster runs, or ledger writes; any new
measurement surfacing from items 2–3 returns as its own [DO].

---

## 6. Paper-grounding measurement list (added 2026-08-18 per the author's row-#121 directive)

The author's ruling (row #121, verbatim there): if option (b) needs a final measurement to
ground it, do it — or at least **collect all measurements we can once the full pipeline is
settled**. "Settled" = the prodcal calibration ladder (`results/pp_coverage_prodcal_20260817/`)
has returned its verdict. The collection, each returning to the author as its own [DO] with a
pre-registered cell:

| # | measurement | grounds | shape / cost class |
|---|---|---|---|
| G-1 | **Catalogue-leg fusion counterfactual** — the item-4 analog with the survival factor fused into the *catalogue* leg (the [P3] fork's direct H₀-impact measurement, superseding the M-4 mixture-share proxy) | decision item 2: makes option (b) "resolved-in-paper" measurable rather than argued; if inert, option (a) becomes the honest pick with a measured bound | paired production counterfactual, 2 venues, item-4 protocol (~170 CPU-h class) OR a harness-scale analog first (~1 CPU-h) to decide whether the production run is warranted |
| G-2 | **Spec-z-kernel σ_z → 0 cell** — one venue cell with the host-z kernel at spectroscopic precision | decision item 3: converts the σ_z→0 no-cost limit from an unmeasured structural argument into a measured statement | single harness/venue cell, ≤1 CPU-h class |
| G-3 | **Gray Eq. (A.10) source re-check** against the published PRD text (UNVERIFIED item iii) | the paper's central quotation | zero-compute read |
| G-4 | **Prodcal ladder verdict itself** (already registered and executing) | the paper's calibration/systematics section: whether the landed configuration is certified at harness fidelity, and the measured sub-term-(b) weight | in flight |

Sequencing of record: ladder verdict (G-4) → G-1-harness + G-2 + G-3 → author picks (a)/(b)
with measurements in hand → figures/text (items 4–5) carry final numbers.

*Append-only from its commit.*

---

## 7. G-3 COLLECTED — 2026-08-18 (appended per the append-only rule)

**Verdict: MATCH.** §4 UNVERIFIED item (iii) is resolved: the repo's rendering of Gray et al.
2020 Eqs. (A.9)/(A.10), the connecting redshift-uncertainty sentence, the absence of any
distance–redshift Jacobian in the A.5–A.10 derivation (the paper's sole "Jacobian" is the
detector/source-frame mass one), and the footnote-3 fragment are verbatim-faithful to
**arXiv:1908.06050v4** (2020-06-12, the version matching the published PRD 101, 122001 (2020);
the PRD page itself is robot-blocked, no known errata affecting Appendix A). Citation of record:
arXiv v4 + PRD DOI. **Caveat for drafting:** the "denominator-only; numerator p_det is the MFG
'most common mistake'" phrasing is a repo gloss consistent with the paper's structure, NOT
Gray's own words — cite as a supported claim, never blockquote. Also corrected: the §4 table's
`L6_DER3_...` path is `results/mechanism_study_20260813/`, not `docs/derivations/`.
