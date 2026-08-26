# CLAIM [P3-MKER] — the with-BH mass kernel's uncertainty budget is incomplete, and the eligibility window has no derivation: "kernel first, window second" (stage 0)

**Opened:** 2026-08-26, author grant (verbatim): *"please open it as suggested by you"* — in
response to the orchestrator's succession proposal for the kernel-zero problem (row #205's
in-fleet exhibit; the proposal's structure is restated in §3 below). Thread tag `[P3-MKER]`
(mass-kernel consistency). **Sequenced AFTER the [P3-2D] verdict** (§5) — the 2D twin's
`mz_sel` object is where this kernel lives; re-deriving the kernel mid-calibration would
invalidate the twin's comparison frame.

## 1. The claim (two parts, both correctness-class, NOT bias-driver-class)

**(a) Kernel:** the with-BH mass likelihood weights candidates by a width dominated by the
GW-conditional σ_cond (production p50 fractional ~1e-8) and does NOT convolve the full
uncertainty budget of the CATALOGUE-side mass: the R&V15 mass-relation intrinsic scatter
(~0.55 dex) is omitted [DOC: [[mass-relation-reines-volonteri]]; 3 related bugs [PHYSICS]-fixed
in 555f018; the log-normal refactor DEFERRED — this thread subsumes that deferral]. A
candidate compatible at ~1σ of its own catalogue error can therefore carry kernel weight
~e^{-k²/2} with k ~ O(10) — physically wrong as a statement about mass compatibility.

**(b) Window:** the eligibility window's k = 1.5 (now symmetric, rows #198–#202) has no
derivation on record (Gate-B row #196: undocumented). It is a physics choice where none
should exist: the window ought to be a TRUNCATION BOUND on the correct kernel — k derived
from a stated tolerance ε (excluded weight < ε of the numerator; Gaussian: k = √(2·ln(N/ε)) on
σ_eff), making the filter an ε-controlled instrument constant with a limiting case (ε→0 ⇒ no
filter ⇒ the exact model).

## 2. Evidence at intake

- **[LOCAL] the in-fleet exhibit (rows #205):** seed 900121 event 20 —
  `L_cat_with_bh = 1.39e-85` with n_sym = 2 window-passed candidates (~1.4σ_g inside the
  window, ~19σ_kernel under the narrow kernel; −176.6 nats). Re-measured this session from
  the fleet artifacts + the zero-compute reconstruction
  (`p3_2d_fleet_20260825/m2link_iii_reattribution_check.json`).
- **[DOC] the scatter omission:** the R&V15 intrinsic scatter (~0.55 dex) omitted from
  host-mass errors (memory + commit 555f018 record; "host-mass errors ~3–7× too tight").
- **[DOC] the window's non-derivation:** Gate-B row #196 (design intent undocumented;
  MATH_REVIEW F5 + IDEALIZATION_LEDGER I4/I7 flag the window, never ratify a k).
- **[DOC] the row-#196 fleet forensic:** ~1.6% of analyzable zeros were the kernel-zero
  class (2/129) — the class this thread names.

## 3. Succession structure (the author-granted proposal, restated)

1. **Kernel first:** derive the convolved with-BH mass kernel — GW width σ_cond ⊕ catalogue
   σ_g ⊕ the mass-relation intrinsic scatter (log-normal, subsuming the deferred refactor) —
   as a 6-item physics-change package (derivation, dimensional analysis, limiting cases:
   scatter→0 recovers the current kernel; σ_cond→0; validity conditions incl. the R&V15
   regime check per the Stage-L assumption register).
2. **Window second:** re-derive k from ε on the ratified kernel's σ_eff — the window ceases
   to be physics.
3. **Measure-first throughout** (the [P3-WBHZERO] pattern transfers): counterfactual flag,
   byte-identical default → mirror-venue paired measurement → production counterfactual
   read → package → author [RULE]. No adoption before measurement.

## 4. Delimitation against the standing exonerations (hard-rule-1 check, PASSED with scope)

Ledger §2 item 1 exonerates the **mass-kernel FAMILY as the 2D-bias driver** (twice: Δ2D
+0.0029 wrong sign #72; 4-cell A/B MAP-unmoved #89; bounded +0.002). **[P3-MKER] does NOT
re-open that claim.** This thread's claim is model-consistency/correctness (the author's
standing values ruling: correctness outranks bias-removal); its H₀ effect may well be small
and is NOT the motivation. Any H₀-effect statement this thread produces must be checked
against that exoneration's bound before banking. Also honored: the honest caveat from intake —
the convolved kernel WIDENS the with-BH channel (σ_M forecast: the with-BH H₀ rescue needs
σ_M ≲ 1–2%), so the correct kernel likely makes the channel more honest AND less informative.

## 5. Sequencing and cheapest decisive measurements

- **HELD behind the [P3-2D] verdict** (the twin calibrates against the current kernel; the
  kernel change re-enters through the twin's own machinery afterwards).
- **Cheapest decisive reads available NOW (zero-compute, rule A1/9):** (i) the fleet-wide
  census of the kernel-zero class (extend the row-#205 scan to all window-passed candidates:
  distribution of kernel-pulls vs window-pulls — quantifies how often σ_window ≫ σ_kernel
  bites); (ii) recompute the 900121:20 kernel value under a convolved σ_eff (analytic, one
  formula) to confirm the exhibit dissolves.
- **Refute by (the claim's own falsifier):** (a) produce a documented derivation showing the
  current narrow kernel is the CORRECT conditional likelihood given the pipeline's
  generative model (i.e., the catalogue mass is treated as exact BY DESIGN elsewhere in the
  chain, consistently in numerator AND normalization) — that would convert (a) from defect
  to design choice; (b) show the σ_M forecast's regime makes the convolved kernel's effect
  numerically indistinguishable (< the twin's ε₂ scale) in EVERY consumer — that would
  demote the thread to documentation-only.

## 6. Stage-L R0 sweep (mandatory at stage 0)

Launched at intake (lightweight): re-read of the already-cited mass-relation and dark-siren
host-weighting papers (R&V15 itself; Gray et al. 2020's host-weighting treatment; the
fastemriwaveforms/EMRI mass-precision references) for stated validity conditions on
catalogue-mass uncertainty treatment. Results append below this line when banked.

---

## R0 SWEEP RESULTS (2026-08-26, [AGENT] sonnet, symptom-card-only; quote-verification per Stage L; banked verbatim summary)

- **[LIT-1, HIGH]** R&V15 §IV.1 (ar5iv full text, quote-verified): "The rms deviation of the
  BH mass measurements from the relation is 0.55 dex, and incorporates both our adopted
  measurement errors of 0.50 dex and a best-fit intrinsic scatter of 0.24 dex (added in
  quadrature)." Sample validity: 262 broad-line AGN, z < 0.055, 10⁸ ≤ M_*/M☉ ≤ 10¹².
  Cross-confirms `docs/MASS_RELATION_ASSESSMENT.md` §2. The single most decisive
  already-cited fact for part (a).
- **[LIT-2, MEDIUM]** Gray 2020 G20-d (already two-fetch-verified in
  `docs/LITERATURE_WARNINGS.md`): host-weighting validated only at 25–75% completeness; our
  venue at 4.79% in-catalogue share — out of the source's validated range.
- **[LIT-3, REPORTABLE ABSENCE]** No cited dark-siren methodology paper (Gray 2020/2023,
  MFG19) treats mass-covariate deconvolution / error-in-variables at all — Gray's
  completeness formalism is magnitude/luminosity-threshold-only. The kernel design must be
  argued from first principles or NEW literature (an R2/R3 ring sweep is the stage-2-time
  follow-up), never assumed literature-compliant.
- **[LIT-4, REPORTABLE ABSENCE]** No cited selection-cut/truncation-bias warning has ever
  been checked against the k = 1.5 hard pre-filter (the symmetric proposal's §3 already
  records the no-derivation fact; the cut-on-observed-vs-cut-on-true question is untouched).
- **Bridge to the Refute-by:** Gray 2023 G23-b (§2.1.3) — truncation/renormalization is
  harmless ONLY under numerator/normalization consistency — is status UNCHECKED against our
  mass-kernel code: checking it IS the §5 Refute-by(a) path (if the current narrow kernel is
  consistently exact-mass on both sides, part (a) demotes to a design choice).

---

## R1 RESULTS [OPUS-ORCH 2026-08-26]

**Stage:** research-cycle stage 1 (measure/refute on the §5 registered zero-compute reads).
**Chair:** Opus orchestration session, 2026-08-26T18:29+02:00, repo HEAD `6e1bc488`.
**Inputs:** two registered reads (§5 (i) census, §5 (ii) exhibit recompute), one code-consistency
recon on the §5 Refute-by(a) / §6 G23-b bridge, one σ_gal provenance trace, and one adversarial
verification pass. **Where a read and the verifier conflict, the verifier governs and the read's
number is recorded as refuted, not reported.** Every decisive number below was additionally
re-derived by this chair from banked artifacts (marked ✓CHAIR); numbers carried from the verifier
without chair re-derivation are marked ✓VER.

**Compute discipline:** zero cluster jobs, zero pipeline executions. All numbers come from reading
banked JSON/CSV and from source-code reads, except read-ii's single ~48 s CPU-only
`pd.read_csv` of `reduced_galaxy_catalogue.csv` (flagged by that read, not hidden; see §R1.7).

---

### R1.1 — THE DECISIVE MEASUREMENT (unregistered, emergent; supersedes both reads' headlines)

The row-#205 exhibit decomposes exactly, from banked artifacts alone. For seed 900121 event 20:

| quantity | bc arm (twin OFF) | bt arm (twin ON) | source |
|---|---|---|---|
| `L_cat_no_bh` | 6.837940436563089e-09 | 6.837940436563089e-09 | `p3_2d_fleet_20260825/b{c,t}_900121_work/seed900121/simulations/diagnostics/event_likelihoods.csv`, row `event_idx=20`, `h=0.73` ✓CHAIR |
| `L_cat_with_bh` | 1.431038452316417e-85 | 1.392199591828355e-85 | ibid. ✓CHAIR (bt value matches §2's cited `1.39e-85`) |
| **total** ln(with/no) | **−176.5606** | **−176.5881** | ✓CHAIR |
| mass-KERNEL log-weight ln(Σ num_w / Σ num_no) over window-passed candidates | **−0.5838** | **−0.6113** | `posteriors_with_bh_mass/h_0_73.json`, `galaxy_likelihoods["20"]`, tuple cols 0 and 2 ✓CHAIR |
| mass-WINDOW log-weight ln(Σ num_no over passed / Σ num_no over all) | **−176.7828** | **−176.7828** (arm-identical) | ibid. + `additional_galaxies_without_bh_mass["20"]` ✓CHAIR |
| retained no-BH numerator fraction | 1.675801e-77 | 1.675801e-77 | ✓CHAIR |

Per-candidate values (bc arm, `galaxy_likelihoods["20"]`, ✓CHAIR):

| catalog_index | num_no | num_w | num_w/num_no |
|---|---|---|---|
| 6791158 | 9.813974564808528e-231 | 3.25213693020771e-231 | 0.331370 |
| 6791138 | 1.0824148618119582e-78 | 6.037718947254922e-79 | 0.557800 |

Window-EXCLUDED complement (`additional_galaxies_without_bh_mass["20"]`, ✓CHAIR):
`6791151` with `num_no = 6.459088535764544e-02` — **~77 orders of magnitude above the best
window-passed candidate's 1.0824e-78** — plus `6791153` at 8.114315798080882e-109.

That `additional_galaxies_without_bh_mass` is the window-EXCLUDED complement is confirmed at
source: `handler.py:663-674` (`candidate_hosts_with_bh_mass = candidate_hosts_without_bh_mass[mass_filter_mask]`)
and `bayesian_statistics.py:4861-4865` ✓VER.

**Attribution.** Two independent routes agree the mass-eligibility WINDOW, not the kernel, carries
the exhibit:
- *residual route* (verifier): total − kernel = **−175.977 nats = 99.67 %**, arm-identical ✓VER.
- *direct-numerator route* (chair): window_ln = **−176.7828 nats**, arm-identical ✓CHAIR.
The two routes differ by 0.81 nats because the raw numerator sums carry no per-host assembly
weights; **that 0.81-nat remainder is UNATTRIBUTED here and is not claimed for either object.**
Under either route the kernel's share is 0.33 % or less. **Deleting the mass kernel entirely would
move the exhibit from −176.56 to ≈ −176.0 nats.**

> **The row-#205 exhibit is a WINDOW-EXCLUSION case, not a kernel-crush case.** §2's
> "~19σ_kernel; −176.6 nats" is not a mass-kernel quantity.

---

### R1.2 — REGISTERED READ §5 (i): the fleet-wide kernel census — DELIVERED

The read initially returned "NOT reconstructible from banked artifacts". **That is refuted:**
`ln(num_w / num_no)` per candidate *is* the mass-kernel log-weight and is fully banked in the
6-tuple (definition at `bayesian_statistics.py:6737-6744` ✓VER). Only the *window*-pull half of
the census needs the 1.68 GB catalogue.

Census over all 24 `bc_*` arms (twin OFF = clean production kernel), `h_0_73.json`,
**independently computed by both the verifier and this chair, agreeing to every reported digit**
(✓CHAIR + ✓VER):

- candidate instances: **2,154,066**; log-scorable **2,122,481**; **31,585 (1.47 %) unloggable**
  (a zero in one channel).
- ln(num_w/num_no): median **−2.9352**, p25 **−4.3494**, p10 **−16.4676**, p01 **−18.2555**,
  min **−21.0053**, max **+0.2301**.
- as k_ub ≡ √(−2 ln mz): median **2.423**, p90 **5.739**, p99 **6.042**, **max 6.482**.
- fraction below −4.5 nats: **0.2373**; below −12.5 nats: **0.1514**; below −50 nats: **0.0000
  (zero candidates fleet-wide)**.

**k_ub is an UPPER BOUND on the mass pull, not the pull** ✓VER: `mz` is a density,
ln mz = −p²/2 − ½ln(2πσ_sum²), so the prefactor contributes negatively whenever σ_sum > 1/√(2π)
= 0.3989 — true here. The exhibit's true mass pulls are 0.305σ and 0.032σ (§R1.3). Every k_equiv
figure in the R1 record is a bound and is labelled as such.

**Event-level Δln proxy** (48 `event_likelihoods.csv`, h = 0.73; verifier reproduced the read's
figures to the digit ✓VER): 4522 rows, 4442 with both channels positive, 80 excluded; median
**1.2351**, p90 **15.0907**, p99 **16.5657**, max **176.5881**; fraction > 4.5 nats **0.18032**
(801), > 12.5 nats **0.16749** (744), > 50 nats **0.00045** (2, both arms of 900121:20).

**Body/tail re-attribution (the operative caveat)** ✓VER: decomposing all 4442 rows,
window_ln has median −0.0003, p10 −0.2642, p01 −1.1374, min −176.7828, while kernel_ln has
median −2.1683, p10 −16.1662, p01 −17.5577, min −19.5590. Among the 904 rows below −4.5 nats the
**median window contribution is exactly 0.0**. So the **BODY of the distribution is genuinely
kernel-driven** (the 18.0 %/16.8 % figures do describe the kernel) while the **extreme TAIL is
entirely window**. The claim card was written about the body; the exhibit was drawn from the tail.
The read's own stated caveat ("a lower bound on individual candidate crush") is superseded by this
one.

---

### R1.3 — REGISTERED READ §5 (ii): the convolved-σ_eff recompute — EXHIBIT DOES NOT DISSOLVE

Production kernel (Gaussian product, `bayesian_statistics.py:6606-6607` + `:6613-6616`,
engaged unconditionally because `normalization_mode` defaults to `generator_marginal`
(`main.py:1373`, `bayesian_statistics.py:6764`) so `resolve_host_mass_kernel` (`:240-300`,
resolution at `:291-295`) returns `gaussian` and `_use_mass_trunc` is False (`:6175-6178`); the
[P3-2D] twin guard at `:6183-6189` *raises* if `mz_sel` is combined with `mass_trunc`, so the
exhibit is **provably** on the Gaussian branch) ✓VER.

Reconstructed inputs for the two window-passed candidates ✓VER (σ_cond re-derived from scratch as
the Schur complement of the 4×4 (φS, qS, d_L/d_L, M/M) block of `prepared_cramer_rao_bounds.csv`
row 20: σ²_cond = 1.0957976761708426e-17, **σ_cond = 3.3102834866078203e-09**; det_M =
1333246.127516857; σ_dL/dL = 0.00505631803897675; σ_φS = 5.706971980093164e-4; σ_qS =
9.971069269674295e-4; SNR = 235.85577567934249. Catalogue side corroborated *without* a catalogue
load by inverting `handler.py:1368-1382` on GLADE-quantised M_* = 0.3 ×1e10 M☉ with σ_M*/M_* = 1.00
and 2.00, reproducing BH_mass = 709540.709 and BH_mass_error = 894866.276 / 1570331.165 exactly):

| candidate | σ_gal_frac | μ_gal_frac | pull (current production) | mz (current) |
|---|---|---|---|---|
| 6791138 | 0.709749 | 0.783444 | **+0.30512 σ** | **0.53652** |
| 6791158 | 1.214813 | 1.039325 | **−0.03237 σ** | **0.32823** |

Convolved recompute, σ²_eff = σ²_cond + σ²_gal_frac + σ²_scatter_frac, with 0.24 dex =
0.5526204 nats and 0.55 dex = 1.2664218 nats (dex→natural-log conversion verified correct — **no
unit error of the class this project has been burned by**) ✓VER:

| candidate | +0.24 dex mz (ratio) | +0.55 dex mz (ratio) |
|---|---|---|
| 6791138 | 0.46384 (**0.8646**) | 0.32191 (**0.6000**) |
| 6791158 | 0.29675 (**0.9041**) | 0.22268 (**0.6784**) |

**Outcome: `exhibit_dissolves = no`.** Convolution moves both candidates' mass-kernel weight
**DOWN 10–40 %, not up** — there was no mass-axis suppression to rescue, both pulls already
< 0.31 σ. **The +0.55-dex ratios (0.6000 / 0.6784) are arithmetically exact but must NOT be banked
as physical magnitudes** ✓VER: the linearisation σ_scatter_frac = σ_ln · μ_gal_frac at σ_ln = 1.27
is precisely the failure mode the codebase documents at `bayesian_statistics.py:456-465`. Only the
DIRECTION (down) is robust.

**Refuted along the way:** read-ii's §5 sky-position lead (19.44 σ) as "the closest match to the
exhibit's −176.6 nats" is refuted, and its "−176.6 was not reproduced anywhere" was a search
failure ✓VER — −176.6 is ln(L_cat_with_bh/L_cat_no_bh) (§R1.1), which read-ii did not form because
it checked ln(L_cat_with_bh) = −195.39 alone and restricted itself to the two window-PASSED
candidates, never inspecting 6791151.

---

### R1.4 — AMENDMENT to §1(a) [not a silent edit; §1 is left standing as written, with this block binding over it]

> **AMENDMENT A-MKER-1 (2026-08-26).** §1(a) mis-states what the code does. All three of its
> factual assertions are refuted at source:
>
> 1. **"does NOT convolve the … CATALOGUE-side mass"** — REFUTED. `bayesian_statistics.py:6607`
>    forms `sigma_gal_frac = host_M_error*(1+z)/_det_M` and `:6613` adds it in quadrature as
>    `sigma2_sum = _sigma2_cond + sigma_gal_frac**2`. The catalogue mass **is** a Gaussian,
>    convolved analytically — in the default branch and in `mass_trunc` alike ✓VER.
> 2. **"a width dominated by the GW-conditional σ_cond (production p50 fractional ~1e-8)"** —
>    REFUTED, with the sign of the domination inverted. For the exhibit,
>    σ_gal_frac / σ_cond = **2.14e8** and **3.67e8**; `sigma2_sum` is dominated by the
>    **CATALOGUE** width to 16 significant figures ✓VER. σ_cond's smallness is what makes the
>    kernel width *equal* the catalogue width, not what makes it narrow.
> 3. **"the R&V15 mass-relation intrinsic scatter (~0.55 dex) is omitted"** — REFUTED as worded.
>    The **0.24-dex intrinsic scatter IS present** as `sigma_int = 0.24*np.log(10)`
>    (`handler.py:41-44`, added in 555f018) and enters the error budget at `handler.py:1375-1381`
>    ✓VER. §6's own banked LIT-1 quote contradicts the premise: R&V15's 0.55 dex is the TOTAL rms
>    = 0.50 dex measurement error ⊕ 0.24 dex intrinsic scatter. Calling the omitted half "the
>    intrinsic scatter" conflates the two halves of the card's own citation.
>
> **Also refuted: "kernel weight ~e^{−k²/2} with k ~ O(10)".** Across 2,122,481 window-passed
> candidate instances the maximum k_ub on the mass axis is **6.482**, and k_ub is itself an upper
> bound (§R1.2). **Not one candidate anywhere in the 24-seed fleet reaches k = 10 on the mass
> axis.** ✓CHAIR + ✓VER.
>
> **What §1(a) becomes, in amended and narrower form:** *R&V15's 0.50-dex virial MEASUREMENT-error
> component is deliberately excluded from `host_M_error`, on a design choice recorded only in a
> code comment (`handler.py:41-42`) and running against `docs/MASS_RELATION_ASSESSMENT.md` §6's
> explicit recommendation to carry the full ~0.55 dex.* Whether a *predictive* error on M_BH given
> M_* should include the calibration sample's measurement error is a live, arguable physics
> question — **not a bug**. That question, and only that question, survives as part (a).
>
> **Magnitude of the surviving question, for scoping** ✓VER: adding the 0.50-dex component raises
> CV² by 1.3255 — **+83 % for 6791138 and +27 % for 6791158**. Real, not order-of-magnitude. Note
> also that read-ii's characterisation of σ_int as the "dominant term" is wrong for these
> candidates: σ_int² = 0.3054 out of CV² = 1.5906 (19 %) and 4.8982 (6 %); the dominant term is
> the propagated GLADE stellar-mass error (β·σ_M*/M_*)², with σ_M*/M_* = 1.00 and 2.00 exactly.
> For these candidates the catalogue-side width is already dominated by GLADE measurement error,
> not by the mass relation at all.

**Correction to §2's exhibit description.** "~19σ_kernel … −176.6 nats" must be read as refuted
(§R1.1). The "18.98" figure was obtained as √(−2·ln(num_w)) from a *dimensionful* redshift-
quadrature result (density × prior × GW likelihood), which is **dimensionally invalid** — the
agreement with "~19" is numerology ✓VER. The correct per-candidate mass-kernel quantity is
ln(num_w/num_no) = −0.5838, i.e. k_ub = 1.08.

**Arm-labelling correction.** Both reads cited "seed 900121 event 20" without naming the arm while
quoting different values (bc: `num_w` = 6.037718947254922e-79; bt: 5.873853242962365e-79). The
bt/bc numerator ratios 0.972907 and 0.972860 match `s4d_at_truth = 0.9729434633257568` in CRB row
20 to 4 digits ✓VER — the twin's S_4D factor behaves as designed and the arms carry the same
injection. **Arms must be named in all future citations of this exhibit.**

---

### R1.5 — REFUTE-BY(a) ADJUDICATION: **SPLIT** — and the falsifier itself **FAILS**

The §5 Refute-by(a) falsifier asked for a derivation showing the narrow kernel is the *correct*
conditional likelihood because catalogue mass is treated as exact BY DESIGN, **consistently in
numerator AND normalization**. Adjudication:

**(A) The falsifier FAILS, decisively, and it fails on a stronger reading than the recon reached.**
The recon returned `INDETERMINATE` on the grounds that two candidate normalization objects
disagree (per-host `D_g`, which mirrors the numerator exactly, vs population-level `Σ_global`,
point-evaluated by default and self-flagged as issue #24 at `bayesian_statistics.py:1654-1657`).
**That INDETERMINATE verdict is refuted: the question is decidable, and neither object is
production's divisor** ✓VER. In `generator_marginal`, `bayesian_statistics.py:5069-5092` computes
`L_cat_with_bh_mass = weighted_sum(num_w) / n_hat_w` with `n_hat_w = self._W_cat / V_f(h)` — a
draw-side conversion the code's own comment (`:5061-5066`) describes as carrying "no P_det inside
the conversion" and being "ONE n_hat_w for both channels". It contains **no mass kernel at all**
and is byte-identical between the no-BH and with-BH channels. The per-host `D_g` is "diagnostic
only in this mode; the assembly never divides by it" (`:6199-6200`) — confirmed: the
`generator_marginal` branch reads `r[2]`, never `r[3]`. The final assembly at `:5544-5545` divides
BOTH channels by the SAME `D_gen` and adds the SAME `B_num`.
> **CONCLUSION: the with-BH mass kernel AND the k = 1.5 mass window both live in the numerator with
> NO mass-side renormalization anywhere in production.** The numerator/normalization pair is NOT
> consistent in the Gray-2023 G23-b sense (§6 bridge). Refute-by(a) therefore **cannot** convert
> part (a) from defect to design choice — the demotion route is closed. The recon's
> `consistent: false` stands; its `INDETERMINATE` and its stated reason are superseded.

**(B) But part (a) as *written* is refuted anyway (A-MKER-1), for reasons independent of (A).**
The code-level framing is wrong; the physics point survives only in the amended, narrower form.

**Verdict: SPLIT.**
- **Code-level framing → REFUTED**, amended in place by A-MKER-1. The card must not be left
  carrying it.
- **Physics point → SURVIVES, re-scoped**, as the 0.50-dex virial-measurement-error question.
- **Design-choice demotion → NOT AVAILABLE**: the registered demotion route (G23-b consistency)
  is itself falsified. A future demotion would need a *different* argument.
- **Part (b) → STRENGTHENED** (§R1.6). The evidence gathered for (a) is (b)'s evidence.

---

### R1.6 — WHAT PART (b) GAINED: σ_window ≫ σ_kernel CONFIRMED AT FLEET SCALE

§5(i)'s stated purpose ("quantifies how often σ_window ≫ σ_kernel bites") is **answered
affirmatively, with a mechanism** ✓VER:

- The window at `handler.py:664-672` compares `(M_z ∓ 1.5·σ_M_gw)` against
  `host_M ∓ 1.5·host_M_error` using the **EVENT's redshift-range bounds** (1+z_max) and (1+z_min).
- The kernel at `bayesian_statistics.py:6606-6607` compares against the **CANDIDATE's own** (1+z).

The two objects therefore use **different redshift conventions**: the window is not a truncation of
the kernel it gates, but a different, far more permissive comparison. It admits candidates the
kernel then suppresses at k_ub up to 6.482 — **23.73 % of candidate instances below −4.5 nats,
15.14 % below −12.5 nats** (§R1.2). This is a genuine, fleet-scale, previously unquantified
finding, and it supports part (b) — *the window ought to be an ε-controlled truncation bound on the
correct kernel* — far more strongly than the exhibit ever supported part (a).

**A second, sharper lever, previously unnamed:** the window multiplies `BH_MASS_ERROR` by
`_bh_mass_error_multiplier` = 1.5 (`handler.py:663-673`). **The same `host_M_error` object feeds
both the kernel and the window.** Inflating it by the 0.50-dex component would therefore *widen the
eligibility window* — potentially readmitting 6791151, where 176 of the exhibit's 176.6 nats live.
**Parts (a) and (b) are coupled through a single quantity.** No prior document records this.

---

### R1.7 — KNOWN vs UNKNOWN, and reportable absences

**NOW KNOWN (banked, sourced, verifier-confirmed):**
1. The row-#205 exhibit is a window-exclusion case; the kernel's share is ≤ 0.33 % (§R1.1).
2. The production kernel is provably the Gaussian branch for this exhibit (§R1.3).
3. σ_cond = 3.3102834866078203e-09 for the exhibit event, re-derived from the CRB Fisher matrix;
   catalogue-side widths dominate by 2.1e8–3.7e8 (§R1.3, §R1.4).
4. The full per-candidate mass-kernel census, 2,122,481 instances, fleet-wide (§R1.2).
5. Production has **no mass-side renormalization**; Refute-by(a) fails (§R1.5).
6. The window/kernel redshift-convention mismatch, and the shared-`host_M_error` coupling (§R1.6).
7. The 0.24-dex intrinsic scatter is present; the 0.50-dex measurement component is not.

**STILL UNKNOWN (and what it would cost):**
1. **`6791151`'s `host_M` and `host_M_error` — NOT FOUND in any banked artifact.** This is *the*
   single measurement that would settle the exhibit. Requires the catalogue load. **Neither read
   attempted it.**
2. **The window-pull half of the census** (|μ_gal − μ_cond| / σ_gal per candidate). Requires the
   catalogue load. Only the kernel half is banked.
3. **The 31,585 unloggable candidate instances (1.47 %)** — a zero in one channel, unenumerable by
   log-ratio. **Neither read enumerated this class**; it is where any *true* kernel zero would have
   to live, and the census is not complete until it is inspected.
4. **σ_cond's fleet-wide p50** (the card's "~1e-8"): NOT re-derived at fleet scale. Only the
   single-event value is measured. The card figure remains claim-card-sourced.
5. **H₀ effect of anything in this record: NOT MEASURED.** Not of the convolved kernel, not of the
   window, not of the 0.50-dex inflation.
6. **Whether the Gray-2023 G23-b bridge is about `D_g` or `Σ_global`:** moot for production
   (neither is the divisor, §R1.5), but the Gray-2023 source text itself was not read — the recon
   was code-only.
7. **Whether the Σ_global point-vs-convolved gap (issue #24, `bayesian_statistics.py:1654-1657`)
   matters in OTHER normalization modes:** not assessed. Instrument J (`:2890-2914`) exists to
   measure it and was not run.

**REPORTABLE ABSENCES (new this stage, additional to §6's LIT-3/LIT-4):**
- **ABS-1:** the exclusion of R&V15's 0.50-dex measurement component has **no derivation document**
  — it is recorded only in a code comment (`handler.py:41-42`), and it contradicts
  `docs/MASS_RELATION_ASSESSMENT.md` §6's explicit written recommendation. An undocumented
  divergence from a documented recommendation.
- **ABS-2:** `host_M_error` is computed **live** at catalogue load
  (`handler.py:1136-1142`, called unconditionally from `__init__:348`) and is **never written to
  any per-candidate banked artifact** in `p3_2d_fleet_20260825/`. Per-candidate catalogue-side
  widths are therefore not auditable from the record without re-running the load. This is a
  reproducibility gap, and it is what made UNKNOWN-1 and UNKNOWN-2 expensive.
- **ABS-3:** the window's `_bh_mass_error_multiplier = 1.5` and the kernel's σ_eff have **never
  been compared in any document** — the coupling in §R1.6 was found only here.

**REFUTED FIGURES — do not carry forward** (recorded so they cannot re-enter): "~19σ_kernel";
"18.98 from √(−2 ln num_w)"; "k ~ O(10)"; "σ_cond dominates"; "the intrinsic scatter is omitted";
"the catalogue mass is not convolved"; the read's "18 arm-instances in 9 seeds" (the cited artifact
`m2link_iii_reattribution_check.json` contains **17** monster events across 9 seeds, 16 in
`predicted_sym_zero`, the non-match being 900121:20 ✓VER — the figure was labelled "exact, banked"
but was not re-derived); the σ_gal read's "measured fractional CV 0.5826–0.5838" as a *production*
figure (those are the zero-stellar-error **floor** at the R&V15 pivot, sampled from the head of the
CSV; the exhibit's own candidates sit at CV = 1.261 and 2.213 ✓VER); read-ii's "σ_int as the
dominant term".

**A branch-label correction for the record** ✓VER: `bayesian_statistics.py:857-860` sets
`narrow = (sigma_gal <= K*sigma_cond) & (sigma_lnM <= 0.1)`; `:861-864` build the **linearised**
`mz_gauss`, selected when the condition **holds**; the exact-lognormal (genuinely convolved) branch
is `mz_gh` at `:852`. The whole crossover lives inside `_mass_trunc_mz_integral` (`:788`),
**unreachable in production**. Counterfactually, at the exhibit's parameters
σ_gal/(K·σ_cond) = 4.29e7 and 7.34e7 (K = 5.0, `:466`) and σ_lnM = 1.261 / 2.213 vs the 0.1 cap
(`:467`), so `narrow` would be FALSE by 7–8 orders of magnitude and `mass_trunc` would take the
exact GH path.

---

### R1.8 — §4 HARD-RULE-1 CHECK: **PASSED**, stated explicitly with the bound

**Nothing in this R1 record re-opens the standing exoneration of the mass-kernel FAMILY as the
2D-bias driver** (Ledger §2 item 1: Δ2D +0.0029 wrong sign, row #72; 4-cell A/B MAP-unmoved,
row #89; **bounded +0.002**). No measurement in §R1.1–§R1.7 is an H₀-effect statement, and none is
offered as one. Specifically:

- The window-exclusion finding (§R1.1, §R1.6) concerns a **different object** — the eligibility
  filter, rows #198–#202's territory — and **its H₀ effect is NOT MEASURED here**. No H₀ claim is
  made about it.
- The census (§R1.2) is a per-candidate likelihood-weight distribution, not a posterior statement.
- **One item to watch downstream:** read-ii's "convolution moves both candidates' mass-kernel weight
  DOWN 10–40 %" is safe as a *per-candidate magnitude*, but if it is ever carried forward as a
  *channel-level* statement it **must** be checked against the +0.002 bound before banking.

---

## R1 INFORMATION FORECAST — stage 1 → stage 2 [OPUS-ORCH 2026-08-26]

*What each candidate measurement would tell us, what it costs, what outcome would change a
decision, and what outcome would not — stated BEFORE spending compute.*

### F-i — §3 item 1: the kernel-first physics-change package. **WARRANTED, but RE-SCOPED and DEMOTED from first position.**

| | |
|---|---|
| **As originally scoped** | "derive the convolved kernel — σ_cond ⊕ σ_g ⊕ the omitted intrinsic scatter" |
| **Status of that scope** | **VOID.** The convolution exists (`:6607`, `:6613`) and the intrinsic scatter is present (`handler.py:44`). There is nothing to add that is not already there. |
| **Re-scoped question** | Should σ(M_BH \| M_*) — a *predictive* error — include R&V15's 0.50-dex virial *measurement* error? A one-line change in `handler.py:41-44`, **not** in `bayesian_statistics.py`. |
| **What it would tell us** | Whether the catalogue-side width is right. Magnitude bounded already: **+83 % / +27 % on CV²** for the exhibit's two candidates (§R1.4) — real, not order-of-magnitude. |
| **Cost** | Derivation + dimensional analysis + limiting cases + validity register + regression test: **zero-compute, ~1 session**. The counterfactual measurement that would follow: a 48-arm fleet re-run (the [P3-2D] fleet's cost). |
| **What outcome would change a decision** | A derivation showing the measurement-error component is *required* for an unbiased predictive density ⇒ `[PHYSICS]` change with a **coupled** window effect (§R1.6). |
| **What outcome would NOT change a decision** | Any kernel-only effect size. The exhibit's kernel share is ≤ 0.33 %; no kernel change can dissolve it. Running the fleet counterfactual **to make the exhibit dissolve** is a guaranteed null and should not be funded on that rationale. |
| **Recommended scope** | Author the derivation (zero-compute). **Do NOT fund the fleet counterfactual at this stage** — its decision value is contingent on F-ii's outcome, because the same `host_M_error` moves both objects. |

### F-ii — §3 item 2: the window ε-derivation. **PROMOTED TO LEAD. Now the thread's centre of gravity.**

| | |
|---|---|
| **What it would tell us** | Whether k = 1.5 can be replaced by k = √(2 ln(N/ε)) on a σ_eff that is *the kernel's own* — converting the filter from physics to an instrument constant with a limiting case (ε→0 ⇒ exact model). |
| **Evidence it now rests on** | (1) the window carries ≥ 99.5 % of the exhibit (§R1.1); (2) the extreme tail of the fleet Δln distribution is entirely window (§R1.2); (3) window and kernel use **different redshift conventions** (§R1.6) — the window is not a truncation of the kernel at all. Item (3) is a correctness-class finding in its own right and is new. |
| **Cost** | The ε-derivation itself is **analytic, zero-compute**. Its *validation* needs the window-pull census (F-iv) and one fleet counterfactual. |
| **Dependency inversion** | An ε-derived k presupposes a σ_eff that both objects share. **F-ii now depends on the σ half of F-i, not the reverse.** The card's "kernel first, window second" is **backwards on this evidence** — but only in *emphasis*: the σ decision still precedes the k decision. Corrected sequencing: **σ-decision → window ε-derivation → kernel derivation → one coupled counterfactual.** |
| **What outcome would change a decision** | An ε-derivation that reproduces k ≈ 1.5 within tolerance ⇒ the window is retroactively justified and part (b) closes as documentation. Anything materially different ⇒ `[PHYSICS]` change to the filter. |
| **What outcome would NOT change a decision** | An ε-derivation on a σ that is *not* the one the kernel uses — that reproduces the current inconsistency in a new notation. |

### F-iii — measurements the reads showed are NOT worth running

1. **A 48-arm fleet counterfactual of the convolved kernel, run to dissolve the exhibit.** **NULL BY
   CONSTRUCTION.** The exhibit's kernel share is ≤ 0.33 % and the convolution moves weight **DOWN**
   (§R1.3). *Cost avoided: one full fleet re-run.*
2. **Instrument J (`sigma4d_mass_kernel="kernel"`, `bayesian_statistics.py:2890-2914`) to close
   issue #24, as this thread's decisive read.** **NOT DECISIVE HERE.** Production's
   `generator_marginal` assembly never divides by `Σ_global`'s per-host mass treatment; the divisor
   is `n_hat_w = W_cat/V_f(h)` (`:5069-5092`), identical in both channels (§R1.5). *Caveat: this
   kills it as **this thread's** read, not globally — it may still matter in other normalization
   modes (UNKNOWN-7).*
3. **Re-running the −50-nats "monster" scan.** **SUPERSEDED.** The per-candidate kernel census
   (§R1.2, zero-compute, now banked) is strictly more informative, and the monster scan's own cited
   count did not reproduce (§R1.7).
4. **A further per-candidate σ_cond fleet census.** **LOW VALUE.** σ_gal dominates by 2.1e8–3.7e8;
   σ_cond's exact distribution cannot change any conclusion in this record.

### F-iv — RECOMMENDED NEXT ACTION (one)

> **Load `reduced_galaxy_catalogue.csv` once, through `GalaxyCatalogueHandler`, and read out
> `host_M` and `host_M_error` for candidate `6791151` (the excluded object carrying 176 of the
> exhibit's 176.6 nats) — and, in the same load, the window-pull half of the census.**

- **Cost:** one CPU-only `pd.read_csv` of **1,681,954,844 bytes / 22,641,048 rows** (~48 s measured
  by read-ii) plus the mapping pass; **≤ 10 min wall** including the script. **No GPU, no cluster,
  no pipeline execution, no simulation.**
- **Why it is decisive:** it is the **only** measurement that settles the exhibit, and neither
  registered read attempted it. Outcomes:
  - **If 6791151 would be readmitted** under a 0.50-dex-inflated σ_gal, or under a
    kernel-consistent (1+z) convention ⇒ the exhibit is a **window defect with a demonstrated fix**,
    parts (a) and (b) fuse into one `[PHYSICS]` package, and F-i's fleet counterfactual becomes
    worth funding.
  - **If it would NOT be readmitted** ⇒ 6791151 is a genuine mass mismatch, the window did the right
    thing for the wrong reason, the exhibit is retired as evidence, and the thread narrows to the
    ε-derivation (F-ii) on documentation grounds.
  - **Either outcome closes UNKNOWN-1 and UNKNOWN-2 and unblocks the sequencing decision.** There is
    no outcome that leaves the thread where it is — which is the definition of a decisive read.
- **Also in scope for the same load (marginal cost ~0):** enumerate the 31,585 unloggable candidate
  instances (UNKNOWN-3) to complete the census.

### F-v — AUTHOR DECISIONS REQUIRED

- **[DO] D-MKER-1 — authorize the catalogue load (F-iv).** This exceeds the thread's zero-compute
  mandate: it is a read of an **un-pinned multi-GB dataset**. Per CLAUDE.md's dataset-pinning rule
  the authorization should carry: *pin the checksum at first read and STOP on mismatch*. This is a
  read-only measurement; it changes no source file.
- **[RULE] R-MKER-1 — ratify AMENDMENT A-MKER-1 (§R1.4).** §1(a) as written is refuted on all three
  factual assertions and stands only in the amended, narrower form (the 0.50-dex virial
  measurement-error question). The card must not be left carrying a refuted code claim.
- **[RULE] R-MKER-2 — ratify the SPLIT adjudication and the closure of the demotion route
  (§R1.5).** Refute-by(a) FAILS: production has no mass-side renormalization, so the G23-b
  consistency route cannot demote part (a) to a design choice.
- **[RULE] R-MKER-3 — rule on the corrected sequencing (§F-ii).** The card's "kernel first, window
  second" is backwards in emphasis; the proposed correction is **σ-decision → window ε-derivation →
  kernel derivation → one coupled counterfactual**. *This is a fresh [RULE]: it rests on evidence
  (§R1.1, §R1.6) that did not exist when §3's structure was granted, and is therefore not covered
  by the intake grant.*

**Uncertainty flagged, not smoothed:** the 0.81-nat gap between the two attribution routes in §R1.1
is unattributed; the +0.55-dex ratios in §R1.3 are arithmetically exact but physically untrustworthy
in magnitude; the fleet σ_cond p50 remains claim-card-sourced; and **no H₀ effect of anything in
this record has been measured** (§R1.8).

*— end R1 [OPUS-ORCH 2026-08-26]*

---

## R2 RESULTS — the decisive catalogue read [OPUS-ORCH 2026-08-26]

**Verdict in one line: NO. Candidate `6791151` is NOT readmitted by the full R&V15 0.55-dex budget.
It misses by a factor of 2.315 on `host_M_error`. The row-#205 exhibit is retired as evidence.**

This section is an ADJUDICATION. Two independent measurers (A: production-loader route; B: direct
chunked-read route) executed the F-iv read. Per the standing rule *"verifier output is evidence, not
authority"*, every decisive number below was **re-derived from source by the chair** on a third,
independent code path before banking. Measurer files: `mker_r2_measure_A.md`, `mker_r2_measure_B.md`.

### R2.0 — Pin discharge (D-MKER-1's condition)

| item | value | source |
|---|---|---|
| file | `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` | — |
| size | **1,681,954,844 bytes** | `ls -l`, this session ✓CHAIR |
| md5 | **`c52c13b5cab61f6b3f04bbe202550969`** | `md5sum`, this session ✓CHAIR |
| cluster copy of record | **BYTE-IDENTICAL** (verified 2026-08-26) | orchestrator pin, carried per CLAUDE.md dataset-pinning rule |
| raw rows | 22,641,048 | chair full pass |
| rows surviving NaN-drop + mass/z prune | **20,834,171** | chair full pass |

No mismatch. D-MKER-1's STOP condition did not trigger. **This was a read-only measurement; no
source file was modified, no simulation, no GPU, no cluster job.**

### R2.1 — Index semantics: VERIFIED, and the cross-check PASSES

`catalog_index` in the banked posteriors is a **0-based positional index into the pruned,
`reset_index()`-ed frame** that `GalaxyCatalogueHandler` builds. It is **not** a raw CSV row number
and **not** a catalogue-native identifier. Chain, at source:

1. `read_reduced_galaxy_catalog` (`handler.py:533-542`) — headerless read, `RangeIndex` = raw row.
2. `_map_stellar_masses_to_BH_masses` (`handler.py:1136-1142`) — overwrites the `STELLAR_MASS` /
   `STELLAR_MASS_ABSOULTE_ERROR` columns **in place** with R&V15 `BH_MASS` / `BH_MASS_ERROR`.
3. `_rotate_equatorial_to_ecliptic` (`:1144-1181`) and `_map_angles_to_spherical_coordinates`
   (`:1183-1199`) — **transform only, no row drops** (chair-verified by reading both bodies).
4. `_remove_galaxies_without_mass_information` (`:1131-1134`) — boolean mask, index-preserving.
5. `_get_pruned_galaxy_catalog` → `_mass_redshift_prune_mask` (`:358-368`, `:215-251`) with
   `M_min=1e4`, `M_max=1e7`, `z_max=1.5` — boolean mask, index-preserving.
6. `setup_galaxy_catalog_balltree` (`:544-556`) — **the single `reset_index()` in the file, line
   555**, which renumbers survivors `0..M-1`.
7. `HostGalaxy.catalog_index = parameters.name` (`:74-81`) — the reset position.

**Cross-check (the task's stop-condition), chair-reproduced exactly:**

| reset position | → raw CSV row (0-based) | `host_M` (M☉) | `host_M_error` (M☉) | required | match |
|---|---|---|---|---|---|
| 6791138 | 7351437 | 709540.708756878 | 894866.2758100418 | 709540.709 / 894866.276 | ✓ |
| 6791158 | 7351457 | 709540.708756878 | 1570331.1654161075 | 709540.709 / 1570331.165 | ✓ |

Both reproduce to full float precision. **`index_crosscheck_passed = true` for A, for B, and for the
chair.** Neither measurer's numbers are void on this ground.

> **A non-trivial hazard that the cross-check earned its keep on:** the local raw-row neighbourhood
> is **not** contiguous — 9 of the 80 raw rows in `[7351400, 7351480)` are pruned (7351413, 7351416,
> 7351417, 7351421, 7351423, 7351426, 7351458, 7351462, 7351464). Position↔row mapping was therefore
> done by a **prune-aware survivor walk anchored on the verified 6791138↔7351437 pair**, never by
> adding a constant offset. An offset-based mapping would have been right here by luck and wrong
> elsewhere.

### R2.2 — Candidate `6791151`, measured

Raw CSV row **7351450** (0-based) = file line 7351451. All columns, exact on-disk names:

| column | value |
|---|---|
| `RIGHT_ASCENSION` (deg) | 98.1045 |
| `DECLINATION` (deg) | −64.2752 |
| `APPARENT_B_MAG` | 19.266656 |
| `REDSHIFT` | 0.052818 |
| `REDSHIFT_MEASUREMENT_ERROR` | 0.0347765573590526 |
| `STELLAR_MASS` (raw, 10¹⁰ M☉) | **0.1** |
| `STELLAR_MASS_ABSOULTE_ERROR` (raw, 10¹⁰ M☉) | **0.1** |
| `REDSHIFT_FLAG` | 1 (photometric) |
| σ_M*/M* | **1.00** |
| derived `BH_MASS` (M☉) | **223 872.11385683485** |
| derived `BH_MASS_ERROR`, current 0.24-dex-only (M☉) | **291 758.99489010876** |
| derived `BH_MASS_ERROR`, full 0.55-dex (M☉) | **389 299.8873277455** |

Derivation is `_empiric_stellar_mass_to_BH_mass_relation` (`handler.py:1368-1382`) verbatim.
**A and B and the chair agree on every one of these to full precision.**

**Variance decomposition of the current budget** (ln² units) — this is what makes the verdict
inevitable, and neither measurer reported it:

| term | value | share |
|---|---|---|
| `sigma_int**2` (the 0.24-dex intrinsic scatter) | 0.30538933116355577 | 18.0 % |
| `d_alpha**2` | 0.033932147907061755 | 2.0 % |
| `(ln(M*/10) * d_beta)**2` | 0.25661186854715445 | 15.1 % |
| **`(beta / M* * sigma_M*)**2`** (propagated GLADE stellar-mass error) | **1.1025** | **64.9 %** |
| **total** | **1.698433347617772** | → σ_ln = **1.3032395587986776** = **0.5660 dex effective** |

The 0.24-dex intrinsic term the card has been arguing about is **not** the dominant term. The
propagated GLADE stellar-mass error is, at 3.6× its variance.

### R2.3 — The window test under the CURRENT budget: CONFIRMED FAIL

GW-side inputs, chair-re-derived (not taken from either measurer):

| quantity | value | source |
|---|---|---|
| `M_z` | 1 333 246.1275168573 | CRB `bt_900121` row 20 (0-based), col `M` |
| `delta_M_delta_M` | 2.691661856213166e-05 | ibid. → `M_z_sigma` = **0.005188122836068134** |
| `luminosity_distance` | 0.2831422160233205 | ibid. → `d_L_uncertainty` = 0.0014316570944745673 |
| `z_min` | **0.05356499027434118** | `get_redshift_outer_bounds` **called directly** ✓CHAIR |
| `z_max` | **0.07776556271743075** | ibid., then `min(z_max, 1.5)` — cap not binding |
| GW floor `(M_z − 1.5·σ)/(1+z_max)` | **1 237 046.5023702232** | ✓CHAIR |
| GW ceiling `(M_z + 1.5·σ)/(1+z_min)` | **1 265 461.692070722** | ✓CHAIR |

`_bh_mass_error_multiplier = 1.5` (symmetric mode, `handler.py:654-661`; confirmed by
`bt_900121_meta.json:24` `"mass_filter_sigma": "symmetric"` and the `sigma_multiplier=1.5` call at
`bayesian_statistics.py:4691`).

`mass_filter_mask` (`handler.py:663-673`), numbers substituted:

```
cond1:  (M_z − 1.5·σ_Mz)/(1+z_max)  ≤  BH_MASS + 1.5·BH_MASS_ERROR
        1 237 046.502370223         ≤  223 872.114 + 1.5 × 291 758.995 = 661 510.606
        →  FALSE                                                    ← BINDING

cond2:  BH_MASS − 1.5·BH_MASS_ERROR ≤  (M_z + 1.5·σ_Mz)/(1+z_min)
        −213 766.378                ≤  1 265 461.692
        →  TRUE                                                     ← not binding
```

`PASS = cond1 AND cond2` → **FAIL. `6791151` is excluded.** The exhibit's premise is confirmed.

### R2.4 — The window test under the FULL 0.55-dex budget: STILL FAILS

**How the inflation is done.** The R&V15 total rms of 0.55 dex is the quadrature sum of a 0.50-dex
virial *measurement* error and the 0.24-dex *intrinsic* scatter (√(0.50² + 0.24²) = 0.5546 ≈ 0.55).
The code carries only the intrinsic component (`handler.py:44`, `sigma_int = 0.24·ln(10)`). Adding
the measurement component means adding `(0.50·ln 10)² = 1.325474…` to the **ln-space variance**
inside the sqrt at `handler.py:1376-1381`:

```
BH_MASS_ERROR_inflated = BH_MASS · sqrt( sigma_int² + d_alpha² + (ln(M*/10)·d_beta)²
                                       + (beta/M*·sigma_M*)²  +  (0.50·ln 10)² )
```

**Reduction check (required by the task):** setting the added component to zero returns
389 299.887… → **291 758.99489010876**, i.e. *bit-identical* to the current value. ✓CHAIR (asserted
in code, `==` True). The inflation is a strict superset of the existing budget.

- σ_ln: 1.3032395587986776 → **1.7389…**, i.e. **0.5660 dex → 0.7552 dex effective**
- `BH_MASS_ERROR`: 291 758.99489010876 → **389 299.8873277455**
- inflation ratio: **×1.3343200865987888**

```
cond1:  1 237 046.502370223  ≤  223 872.114 + 1.5 × 389 299.887 = 807 821.945
        →  FALSE   (still)                                          ← BINDING
cond2:  −360 077.717         ≤  1 265 461.692   →  TRUE             ← not binding
```

> ### **`6791151` is NOT readmitted. The answer is NO.**

### R2.5 — THE MARGIN (the real deliverable)

Solving `cond1` at equality for the error:

```
BH_MASS_ERROR_required = (GW_floor − BH_MASS) / 1.5
                       = (1 237 046.5023702232 − 223 872.11385683485) / 1.5
                       = 675 449.5923422589  M☉
```

| comparison | factor |
|---|---|
| required / **current** (0.24-dex-only) | **× 2.3150943215878144** |
| required / **full 0.55-dex** | **× 1.7350367013428094** |
| fraction of the gap the 0.50-dex measurement term actually supplies | **25.42 %** |
| required effective σ on M_BH | σ_ln = 3.0171225022434265 → **1.3103 dex** |
| extra quadrature term needed on top of the current budget | **1.1818 dex** |
| central-mass shortfall with *no* error term at all | `GW_floor / BH_MASS` = **× 5.52568375336514** |

**Read this plainly:** the full published R&V15 budget closes about **one quarter** of the gap.
Readmission needs an effective **1.31 dex** of M_BH uncertainty — **2.4× the entire R&V15 published
total** — for which no citation exists. This is not a near miss. It is not rescuable by any defensible
widening of the mass-error budget.

### R2.6 — Physics sanity check: is a 76-orders-dominant candidate really excluded on mass?

Yes — and the chair independently reproduced the **entire filter chain** for event 20 to prove the
index is not mis-read. Cone radius (chord, `handler.py:627-633`) = **0.0014956979545757095**:

| position | raw row | sky chord | in cone? | z-filter | `BH_MASS` | `BH_MASS_ERROR` | mass window |
|---|---|---|---|---|---|---|---|
| 6791134 **(true injected host)** | 7351433 | 1.674660e-03 | **NO** | pass | 709 540.709 | 1 110 176.710 | (would pass) |
| 6791138 | 7351437 | 1.335550e-03 | yes | pass | 709 540.709 | 894 866.276 | **PASS** |
| 6791151 | 7351450 | 5.275341e-04 | yes | pass | 223 872.114 | 291 758.995 | **FAIL** |
| 6791153 | 7351452 | 9.050538e-04 | yes | pass | 223 872.114 | 291 758.995 | **FAIL** |
| 6791158 | 7351457 | 1.488549e-03 | yes | pass | 709 540.709 | 1 570 331.165 | **PASS** |

The reproduced cone is **exactly** `{6791138, 6791151, 6791153, 6791158}`; the reproduced pass-set is
**exactly** `{6791138, 6791158}` and the reproduced exclusion-set **exactly** `{6791151, 6791153}` —
matching the banked `galaxy_likelihoods["20"]` and `additional_galaxies_without_bh_mass["20"]`
member-for-member. **The index is not mis-read and there is no units error.** This also closes
Measurer B's open caveat (B did not verify the sky filter): `6791151` **is** in the cone, so it does
reach the mass filter and is excluded there.

The exclusion is physically coherent: `6791151` has a raw stellar mass of 0.1 (10¹⁰ M☉) against
`6791138`/`6791158`'s 0.3, giving it a central M_BH **5.53× below** the GW-required floor. Its huge
no-BH numerator comes from its **redshift**, which the no-BH likelihood is built on and which is a
good match; its mass is not. **That is a genuine mass mismatch, and the window did exactly what a
mass window is for.**

### R2.7 — Two findings the chair adds, which neither measurer reported

**(i) The true injected host is OUTSIDE the sky cone for this event.** `host_galaxy_index = 6791134`
(CRB row 20) sits at chord 1.674660e-03 against a cone radius of 1.4956979545757095e-03 — a factor
**1.1196** outside, i.e. it would need `sigma_multiplier ≈ 1.679` instead of 1.5 to be captured. It
*would* have passed the mass window had it reached it. **The headline exhibit's entire candidate list
therefore contains no true host — it is built on interlopers.** This is a host-recovery observation,
not a mass-kernel one, and it is a further independent reason the exhibit is a poor headline.

**(ii) The window's functional form is linear while its error model is log-normal.** R&V15 is a
log₁₀-linear relation with dex scatter, so `BH_MASS_ERROR` is a *log-normal* width
(σ_ln = 1.3032, i.e. a factor e^1.3032 = 3.68 per σ). `mass_filter_mask` applies it as a **linear
symmetric ±1.5σ interval**. Consequences, arithmetic only:

- the lower edge is **negative** (−213 766 M☉ for `6791151`; −955 724 M☉ for `6791134`) — an
  unphysical mass, which is why `cond2` is never binding for low-mass hosts;
- the upper edge reaches only `1 + 1.5·σ_ln` = **2.955×** the central mass, whereas a log-space
  1.5σ edge would reach `exp(1.5·σ_ln)` = **7.06×**;
- in ln-space `6791151` sits **1.3117 σ** below the GW floor — *inside* 1.5σ. Under a log-space
  window with the **current, unmodified** 0.24-dex budget its upper edge would be **1 581 192 M☉ ≥
  1 237 047 M☉ → it would be READMITTED.**

**This is stated as an observation, not a recommendation, and it cuts both ways.** The object a
log-space window would readmit is `6791151` — an interloper, **not** the true host (which is lost
upstream to the sky cone regardless). Widening or re-shaping the window would therefore restore a
dominant-weight **non-host** to the with-BH list. Whether that is more or less correct is a physics
question the chair does not adjudicate. It is registered here because it is the *only* construction
found that flips this exhibit, and because it means **the window's geometry, not its width, is the
live question** — which sharpens F-ii's ε-derivation onto a concrete target.

### R2.8 — Measurer adjudication

**A and B agree on every decisive quantity** (`host_M`, `host_M_error` current and inflated, both
window tests, the verdict, and the margin factor), and the chair's independent re-derivation
confirms all of them to full float precision. No decisive disagreement exists. Three
non-decisive discrepancies, reported rather than smoothed:

1. **Measurer A's "current budget ~0.738 dex effective" is NOT reproducible and is wrong.** The
   correct value is **0.5660 dex** (= σ_ln/ln 10 = 1.3032395587986776 / 2.302585…). A's *other* dex
   figure ("readmission requires ~1.310 dex effective") is correct and matches the chair exactly.
   The error is confined to one descriptive line and touches no decisive number.
2. **B's "total budget ~1.20 dex" uses a different definition than A's "~1.310 dex"** — B quotes the
   *scatter component only* (√(0.24² + 1.1818²)), A quotes the *effective total on M_BH*. Both are
   internally consistent and both correspond to the same required `BH_MASS_ERROR` = 675 449.592.
   **The unambiguous statement is R2.5's: effective 1.3103 dex, extra term 1.1818 dex.**
3. **Both measurers independently flagged, and the chair independently confirms, a latent code
   defect in `get_redshift_outer_bounds` (`physical_relations.py:546-567`):** the function's own
   `sigma_multiplier` parameter is **dead code** — the body hardcodes `3 *` at lines 563 and 566 —
   so the `sigma_multiplier=2.0` passed at `bayesian_statistics.py:4676` has no effect and the
   bounds are 3σ, not 2σ. `Omega_m_min`/`Omega_m_max` are likewise computed into
   `Omega_de_min`/`Omega_de_max` and then never used. **This does not affect this verdict** (the
   chair called the production function directly, so the measured `z_min`/`z_max` are the ones
   production actually uses), but it is a real defect and is escalated in R2.10.

### R2.9 — What the NO verdict means for the thread's two parts

**Part (a) — the omitted 0.50-dex virial measurement component.** The proposed **fusion does not
happen.** The amended §1(a) question survives *only* as a standalone modelling question: the code
excludes a component that `docs/MASS_RELATION_ASSESSMENT.md` §6 explicitly recommends including, on
a design choice recorded **only** in a code comment (`handler.py:41-42`). That remains a live
documentation/modelling issue on its own merits — but R2 removes its claimed *consequence*. It is
**not** demonstrated to change any outcome in this exhibit, and it supplies only 25.4 % of the gap.
Its priority should drop accordingly.

**Part (b) — the eligibility window.** **The exhibit is RETIRED as evidence.** Seed 900121 / event 20
cannot be cited as a window defect: the window excluded a genuine mass mismatch (5.53× low in central
mass), it retained the only two cone members that were mass-compatible, and the object it excluded is
not the true host. R2.7(i) adds that the true host was already lost to the **sky** filter, so this
event demonstrates nothing about the mass window's correctness either way. The thread narrows to
**F-ii's ε-derivation on documentation grounds** — now with the sharper target from R2.7(ii): the
open question is the window's **linear-vs-log geometry**, not its width.

**Nothing in R2 measures an H₀ effect.** §R1.8's caveat stands unchanged.

### R2.10 — Honest caveats carried

1. **Single candidate, single event.** This is seed 900121 / event 20 only. It does **not**
   establish that the 0.50-dex component would fail to readmit window-excluded candidates elsewhere
   in the fleet. A fleet-level statement would need F-i.
2. **The quadrature convention was applied as instructed, not derived.** Combining the 0.24-dex
   intrinsic and 0.50-dex measurement components in quadrature in ln-space is the natural reading of
   R&V15 §4.1, but the chair did **not** re-derive that this is the physically correct combination.
   That derivation belongs to `/physics-change` if the lever is ever pursued. The verdict is robust
   to it: even the ×1.334 inflation is 4× short of what is required.
3. **`M_z_sigma` taken as banked.** `delta_M_delta_M = 2.691661856213166e-05` on a 1.33e6 M☉ mass is
   a fractional uncertainty of ~3.9e-9. It was **not** re-derived from the Fisher matrix. It is also
   **irrelevant to this verdict**: at 1.5σ it moves the GW floor by 0.0078 M☉ out of 1 237 046.5, a
   relative effect of ~6e-9. The window here is governed entirely by the (1+z) stretch and the
   candidate's own `BH_MASS_ERROR`.
4. **The R2.7(ii) log-space construction is arithmetic on the current code, not a proposal.** It has
   not been pre-registered, not derived, and not run through `/physics-change`. It is banked as a
   registered observation so that it cannot be re-discovered and presented later as a new finding.
5. **The 0.81-nat unattributed gap from §R1.1 is untouched by R2** and remains unattributed.

### R2.11 — AUTHOR DECISIONS REQUIRED

- **[RULE] R-MKER-4 — ratify the NO verdict and the RETIREMENT of the row-#205 exhibit.**
  `6791151` is not readmitted at the full 0.55-dex budget; the margin is ×2.315 on `host_M_error`
  (×1.735 beyond the full budget). Seed 900121 / event 20 must no longer be cited as evidence for a
  mass-window defect, in this card or downstream. *Fresh [RULE]: rests on evidence created by R2.*
- **[RULE] R-MKER-5 — rule on the standing of part (a) after de-fusion.** The 0.50-dex omission is
  now a documentation/modelling question with **no demonstrated consequence**. The author's call:
  keep it open at reduced priority, or close it as a documented design choice with the comment at
  `handler.py:41-42` promoted into `docs/`. *Fresh [RULE].*
- **[DO] D-MKER-2 — authorize pre-registration of the window-GEOMETRY measurement (R2.7(ii)).**
  Linear-symmetric window vs. log-normal error model, including the negative-lower-edge symptom.
  This is a **pre-registration authorization only** — not a `[PHYSICS]` change, and explicitly not a
  recommendation to widen the window (it would readmit an interloper, not the true host).
- **[DO] D-MKER-3 — file the `get_redshift_outer_bounds` defect (R2.8 item 3) as a GitHub issue**
  (`bug`, `physics`): dead `sigma_multiplier` parameter (hardcoded `3 *`, `physical_relations.py:563,566`)
  and unused `Omega_de_min`/`Omega_de_max`. Latent, affects every event's `z_min`/`z_max`, and is
  **independent of this thread**. No fix proposed here.
- **[RULE] R-MKER-6 — rule on whether R2.7(i) (true host outside the 1.5σ sky cone for event 20)
  opens a host-recovery thread.** The chair takes no position on scope. *Fresh [RULE].*

**Sequencing note.** R-MKER-3 (§F-v) proposed *σ-decision → window ε-derivation → kernel derivation →
one coupled counterfactual*. R2 **removes the σ-decision from the critical path** (part (a) is
de-fused and consequence-free here) and **replaces the window ε-question's content** (geometry, not
width). If R-MKER-4 is ratified, R-MKER-3 should be re-stated before it is ruled on.

*— end R2 [OPUS-ORCH 2026-08-26]*
