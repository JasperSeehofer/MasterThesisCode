# BOOK_DESIGN.md — the single authoritative build spec

**Status: BINDING for the chapter-build fan-out.** Written 2026-07-31 by the design
synthesizer, merging the three phase-2/3 documents:

- `BOOK_SOURCES_MAP.md` — concept graph B1–X6, trust tiers, conflict register (§7), museum table
- `BOOK_PEDAGOGY.md` — personas, discovery arc, 60 questions, 21 traps, interactive specs
- `BOOK_TECH_DESIGN.md` — static-site architecture, vendored libs, generator pipeline, CI

Where this file and those disagree, **this file wins** (it was written last and resolves the
merges). Where this file is silent, those files are the spec — in particular the **full
question texts and answers live in `BOOK_PEDAGOGY.md` Part 3** and are used verbatim, and the
**conflict/supersession register in `BOOK_SOURCES_MAP.md` §7 is binding in its entirety**.

Repo paths: source repo = `/home/jasper/Repositories/MasterThesisCode` (READ-ONLY);
worktree = `/home/jasper/Repositories/MasterThesisCode-book` (all writes). Python =
the source repo's `.venv/bin/python`.

---

## 0. The merge decision (chapter partitions reconciled)

The two proposals were: sources-map §6 (11 chapters + 2 museum interludes) and pedagogy
Part 2 (Ch 0–11 + museum annex + 5 interludes). **The pedagogy arc is adopted as final** —
it is a strict refinement of the graph partition with three deliberate improvements the
synthesizer confirms:

1. **Fisher/CRB deferred to Ch 6** (after the mixture, immediately before the z-kernel),
   so M4's "z-broadening exposes the mass defect" lands with K3 fresh — the sources map
   itself requires Ch 6 → Ch 7 adjacency.
2. **Normalization modes + C9 folded into Ch 9** (the forward-model chapter) — the estimand
   question only makes sense once the generator is on the table.
3. **Five short museum interludes carried inside chapters** (each inoculating the next
   chapter's tempting mistake) **plus one annex page** holding all exhibits — supersedes the
   sources map's two standalone interlude slots.

Concept-coverage check (every node B1–X6 has exactly one home chapter; preview mentions
allowed only via forward-reference chips):

| Node group | Home |
|---|---|
| B1, B2 | Ch 2 |
| B3, B4, B5, E1 (intro level) | Ch 1 |
| E2, E3, E4 | Ch 6 |
| H1, H2, H3 | Ch 3 |
| S1, S2, S3, S4 | Ch 4 |
| C1, C2, C3, C4 | Ch 5 |
| K1, K2, K3, K4, K5 (K6 preview only) | Ch 7 |
| M1, M2, M3, M4, M6 (M5 introduced) | Ch 8 |
| N1, N2, N3, N4, N5, N6 · R2, R3, R4, R5 · plunge window | Ch 9 |
| V1, V2, V3, V4, V5, V6 · R1 · X4 | Ch 10 |
| X1, X2, X3, X5, X6 · K6, M5, N6 (adjudicated) · R6 | Ch 11 |
| Museum table #1–#12 + #49a + archaeology #49b | Museum annex + 5 interludes |

Total: **prologue (Ch 0) + 10 build chapters (Ch 1–10) + honest closing (Ch 11) + museum
annex** — within the mandated 8–12 + prologue + closing envelope. The safe-merge fallbacks
(1+2, 9-into-10) are recorded but **not exercised**; do not merge 6+7 or 4+5 under any
revision.

---

## 1. Final chapter list

Formats used below —
**Sources**: `artifact §` per section (chapter agents cite these and only these; line
numbers are anchors to re-grep). **Interactives**: id · manipulate · data source
(`gen` = chapter generator JSON; `js` = closed-form in-browser, must carry `prov-chip toy`;
`rec` = recorded historical measurement presented as data, chipped with its artifact) ·
the AHA · static fallback. **Questions**: verbatim from `BOOK_PEDAGOGY.md` Part 3.
Bias-rail state = what `Book.biasRail` shows on this page.

---

### Ch 0 — Two Numbers That Should Be One  *(Prologue)*
`ch00-two-numbers.html` · `gen_ch00.py` (optional — may be generator-free) · ~1,200 words

- **Discovery statement:** every rung of the local ladder is calibrated against another
  rung — there is no independent ruler.
- **Learning goals:** (i) the H₀ tension as two precise, incompatible measurements;
  (ii) the difference between *a number* and *a measurement with a stated failure mode*;
  (iii) the book's contract: we will build an estimator and spend most of the book trying
  to break it; (iv) why arbitration needs uncorrelated systematics, not just precision.
- **Opening hook:** two bands (Planck-like, SH0ES-like) that do not overlap, drawn before
  a single equation. "Which one is wrong?" is the wrong question.
- **Sections:** 1. Two numbers (context; no project claims). 2. What would arbitrate
  (precision vs systematics budget). 3. The contract of this book (the build-and-break
  arc; the bias-ledger rail is introduced empty).
- **Sources:** framing only — `BOOK_PEDAGOGY.md` Ch 0 card; the h = H₀/100 convention from
  `constants.py` (`H=0.73` as the *mock truth*, stated as such). No pipeline claims are made
  in this chapter, so no pipeline citations are needed.
- **Running example:** none yet (dossier opens in Ch 1). The chapter *names* the promise:
  "one event will follow us the whole way."
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I0.1 Arbitration Budget | σ and unknown-systematic sliders of a hypothetical third method | js (toy) | precision without systematics control arbitrates nothing | static: three example (σ, sys) points with verdicts |
- **Questions:** Q0.1–Q0.3 (+ Trap 0.A) — `BOOK_PEDAGOGY.md` Part 3 §Ch 0.
- **Bias rail:** rendered, single row "estimator: not defined yet".
- **Museum interlude:** none.

---

### Ch 1 — A Ruler That Needs No Ladder
`ch01-ruler.html` · `gen_ch01.py` · ~2,000 words

- **Discovery statement:** GR gives d_L absolutely — and says nothing at all about z.
- **Learning goals:** amplitude ∝ 1/d_L with a known constant; EMRI anatomy (M, μ, ~10⁵
  cycles, plunge inside mission window); phasing carries masses, amplitude+geometry carry
  distance; the dark-siren gap defined; d_L(z; h, Ω_m) and why the fiducial cosmology is a
  *choice* (Barausse M1 consistency, Planck mismatch quoted not absorbed).
- **Opening hook:** the ladder's rungs each calibrated on the previous one (from Ch 0) vs
  one waveform whose absolute scale is GR.
- **Sections:** 1. The waveform as a ruler (E1 intro: amplitude/phasing split).
  2. Meet the EMRI (parameters, event definition = plunge in mission span — sidebar only).
  3. The missing scale: d_L(z; h, Ω_m) (B4). 4. Whose universe? The fiducial choice (B5).
  5. The gap, stated (failure planted: no z for a source with no light).
- **Sources:** B3/B4 — `derivations/dark_siren_likelihood.md` §2.4;
  `physical_relations.py:132 dist`, `:447 dist_to_redshift`; wCDM guard GitHub #4 /
  `physical_relations.py:36`. B5 — `docs/gates/G7_systematics_budget.md` row 6, GATE_SIGNOFF
  G11, ledger #53, #59 (Ω_m-era non-effect, wrong-signed −0.00059). E1 —
  `parameter_estimation.py:335, :488`. Plunge-window sidebar —
  `docs/derivations/plunge_window_initial_conditions.md` (RATIFIED 2026-07-28; sidebar at
  most, per sources map §8).
- **Running example:** **the dossier opens** — EMRI-889 (M = 7.25×10⁵ M☉, μ = 10 M☉,
  d_L = 88.9 Mpc, SNR 1425, σ_dL/dL = 8.0×10⁻⁵, from
  `seed61000/real_r1/prepared_cramer_rao_bounds.csv`) with the empty slot **z: unknown**.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I1.1 Amplitude, Phase, Distance | d_L, inclination, sky position of a toy EMRI strain | js (toy, chipped) | distance comes from the *joint* fit — inclination mimics amplitude | static: two waveforms, same amplitude, different (d_L, ι) |
  | I1.2 Meet EMRI-889 | open the dossier card | gen (CRB row) | the empty z slot *is* the book | inherently static — allowed by spec |
  | I1.3 d_L(z; h, Ω_m) explorer | h and Ω_m sliders on the distance–redshift curve | js (closed-form E(z) integral; validated against `dist()` values baked into the JSON by gen) | h and Ω_m bend the curve differently; at low z only h matters | static: curve family at 3 h values |
- **Questions:** Q1.1–Q1.4 (+ Trap 1.A) — `BOOK_PEDAGOGY.md` §Ch 1.
- **Bias rail:** "estimator: not defined yet".
- **Museum interlude:** none.

---

### Ch 2 — Bayes, Once and For All
`ch02-bayes.html` · `gen_ch02.py` · ~2,200 words

- **Discovery statement:** one distance and no redshift is not yet an inference; and N
  events do not sharpen like √N.
- **Learning goals:** posterior over the single scalar h on a grid; log-additivity;
  MAP vs mean; pull; bias/scatter/coverage as *independent* failure modes; information
  concentration (76/1588 carry 100%, 3 carry 46%).
- **Opening hook:** "I have one event with d_L and no z" is not yet an inference.
- **Sections:** 1. Bayes over h (B1; grid + log-space, citing the production combiner).
  2. Stacking events (B2 preview of marginalization as the missing piece).
  3. A vocabulary for "wrong" (bias, scatter, coverage, pull). 4. The lurch (information
  concentration; failure planted: to get L_i(h) you need a redshift → tens of thousands of
  candidates).
- **Sources:** B1 — `derivations/dark_siren_likelihood.md` §1.1;
  `bayesian_statistics.py:1954 evaluate()`; `posterior_combination.py`
  (`combine_log_space`, physics-floor; Loredo 2004, Mandel–Farr–Gair 2019 arXiv:1809.02063).
  Information split — `IDEALIZED_BASELINE_READOUT.md:42-47`. Grid rationale — pedagogy
  Q2.4 (zoom-grid σ_h = 0.00030 vs 0.005 step).
- **Running example:** 889 revealed as one of the 3 golden events carrying 46% of the
  idealized constraint; its single-event 1D likelihood row is plotted for the first time.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I2.1 Event Stacker | N slider; order toggle (random / sorted by SNR) | gen (real_r1 posteriors + SNR from CRB csv; combination via `Book.combineLogRows`) | the lurch: hundreds of events barely move it, then one golden event lurches it | static: posterior at N = 10/100/1588 with the lurch annotated |
  | I2.2 Bias/Scatter/Coverage Trainer | true-bias, true-σ, stated-σ sliders; 200 seeded repeats | js (toy, chipped) | an unbiased estimator can fail coverage catastrophically | static: three panels, one per failure mode |
- **Questions:** Q2.1–Q2.5 (+ Traps 2.A, 2.B) — `BOOK_PEDAGOGY.md` §Ch 2.
- **Bias rail:** "estimator: not defined yet" (the rail explains it will light up in Ch 4).
- **Museum interlude:** none.
- **Note:** this chapter supersedes the ch00 pipeline demo; the agent may lift code from
  `gen_ch00_demo.py` / `ch00-demo.html` but owns fresh files.

---

### Ch 3 — Which Galaxy?
`ch03-which-galaxy.html` · `gen_ch03.py` · ~2,500 words

- **Discovery statement:** not knowing the host is not an error bar — it's a sum, and the
  weights in that sum are astrophysics I have to commit to.
- **Learning goals:** marginalizing a discrete latent; the candidate ball (BallTree, z
  window; **1D never sees the mass window** — `handler.py:592` vs `:605`, load-bearing for
  Ch 8); rate weight w_g = R_eff(M_g)/(1+z_g) and generator/inference weight identity;
  L_cat as **ratio of sums** (Gray A.9/A.10 via G2c); the h-sweep mechanism ("how much
  galaxy is where this h says the source must be").
- **Opening hook:** the localization ball holds tens of thousands of galaxies and you
  cannot tell which is the host.
- **Sections:** 1. The candidate ball (H1). 2. Who hosts an EMRI? (H2; w_g; the D1
  documented approximation — w evaluated at z_g, outside the ∫dz, honest and open).
  3. The ratio of sums (H3; mean-of-ratios museum preview). 4. The mechanism (h sweeps the
  shell across different galaxies — "nothing else in the book matters if the reader misses
  this"). 5. Failure planted: run it → MAP 0.60.
- **Sources:** H1 — `galaxy_catalogue/handler.py:505, :519, :584-592, :594-603`.
  H2 — `G2c_gray_a9_a10_mapping.md` §2 row w_g + deviation D1;
  `bayesian_statistics.py:879 _rate_weight`; Babak et al. 2017 arXiv:1703.09722;
  `handler.py:765 draw_rate_weighted_hosts`. H3 — `G2c` §2 rows N_g/D_g/
  `weighted_ratio_of_sums`, §4; `bayesian_statistics.py:804, :855, :3532, :3722, :3742`;
  ratio-of-sums fix `816f904`, ledger #26 (1D 0.750 → 0.740). Frame-fix history — ledger
  #12 (cited, detail deferred to Ch 6).
- **Running example:** 889's actual candidate ball around host index 859360; the reader
  places a marker on their guessed host (persisted; re-surfaced in Ch 11).
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I3.1 Sky-Ball Explorer | h slider (+ optional ball-size controls); place-your-marker predict | gen (GLADE+ patch sample + candidate sets per h-grid point for one event; positions/z from the local reduced catalogue — safe: only `z_error` is stale, sources map §7.19d) | sweeping h sweeps the shell across a *different galaxy population*; the reader's marker is usually wrong | static: shell at h = 0.65/0.73/0.81 over the same patch |
  | I3.2 Ratio of Sums vs Sum of Ratios | form toggle on a candidate set | js (toy candidate set, chipped) + rec (the measured 0.750 → 0.740, ledger #26, overlaid) | an "obviously equivalent" rearrangement is worth 0.010 in h | static: the two curves + the measured Δ |
- **Questions:** Q3.1–Q3.5 (+ Traps 3.A, 3.B) — `BOOK_PEDAGOGY.md` §Ch 3.
- **Bias rail:** "estimator: not defined yet" → last section flips it to the first live
  entry: "cat-only, no selection: MAP 0.60 (bias −17.8%)" `H0R:1980` / ledger #9.
- **Museum interlude:** none (the mean-of-ratios story is told in-chapter; full exhibit in annex).

---

### Ch 4 — The Universe Only Shows You Its Loud Half  *(the pedagogical hinge)*
`ch04-loud-half.html` · `gen_ch04.py` · ~2,500 words + interlude ~400

- **Discovery statement:** if I don't divide by what I could have seen, my posterior
  doesn't get noisy — it runs to the edge of the prior and stops caring what the truth is.
- **Learning goals:** conditioning on detection (MFG 2019); p_det as an h-invariant
  horizon-survival estimator; D(h) full-volume vs local-window (MAP 0.60 → 0.73, bias
  −17.8% → 0.0%); the "counted exactly once" principle, named here for the whole book.
- **Opening hook (cold open, the book's starkest):** run Ch 3's estimator on real data —
  the posterior peaks at **0.60, the bottom of the prior**.
- **Sections:** 1. What did we condition on? (S1). 2. p_det from injections (S2; the
  elegant h-invariance of the horizon set). 3. D(h), and why full-volume (S3; the
  docstring's wrong Gray citation flagged per G2c §6 C2 — cite "denominator of (A14)").
  4. Counted exactly once (S4; double-counting history `6754ddb`; starvation-overturn
  told in one paragraph, full exhibit in annex). 5. Failure planted: 95% of events have
  no catalogued host at all.
- **Sources:** S1 — MFG 2019 arXiv:1809.02063; `G2c` D3; Phase 32 / ledger #9 / `H0R:1980`.
  S2 — `simulation_detection_probability.py` module docstring (Finn & Chernoff 1993;
  Finn 1996); injection pool `gate_b_20260730/injection_pool_mix200k_20260728` (200,807
  rows, z_cut 1.5, fingerprint `dl_max(0.73)=9.164987 Gpc`); defect shelf ledger #3, #8,
  #16, #21, #33. S3 — `G2c` §2 row D(h) + §6 C2;
  `bayesian_statistics.py:1013, :1170`. S4 — `G2b` §3.2; `G6_starvation_postmortem.md:24-33`;
  ledger #20, #15/#17, #41→#52.
- **Running example:** 889 (SNR 1425) sails over the threshold — the point is the events
  we *never see*; 889's detection horizon drawn on the population.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I4.1 Delete the Denominator | 3-way switch D(h): off / local window / full volume; predict-marker first | gen (off/full reconstructed from `event_likelihoods.csv` via D = B_num/((1−w_G)·L_comp); local = rec overlay, Phase 32 numbers) | the failure mode is a *rail*, not a nudge | static: three posteriors, MAPs 0.60 / 0.60 / 0.73 annotated |
  | I4.2 Horizon Breather | h slider | gen (survival curve + D(h) from the injection pool) | D(h) is a *visible volume*, not an abstract normalization | static: p_det(d_L) + D(h) curves |
- **Questions:** Q4.1–Q4.5 (+ Traps 4.A, 4.B) — `BOOK_PEDAGOGY.md` §Ch 4.
- **Bias rail:** the flagship movement — flips live between −0.178 and 0.000 with I4.1's
  switch (the rail's `active` entry follows the toggle).
- **Museum interlude (end of page): "The p_det anchor"** — ledger #17: +12% anchor lift
  moved the MAP **zero grid steps**. Lesson: the layer you can see is not the layer that
  matters. Links annex exhibit `museum.html#ex-pdet-anchor`.

---

### Ch 5 — The Galaxy You Cannot See
`ch05-unseen-galaxy.html` · `gen_ch05.py` · ~2,800 words + interlude ~400

- **Discovery statement:** for 1512 of 1588 events the host is not in the catalogue at all.
- **Learning goals:** completeness f(z,Ω); the binary latent G/Ḡ; the master equation
  p_i = w_G L_cat + (1−w_G) L_comp with w_G = β_G/D (verified identity, 2.8e-16);
  B_num and the 4π sky marginal (the 5000× defect); the zero-host fallback (real bug,
  fixed, symptom unmoved); **the splinter**: class argmaxes 0.86 (in-cat) vs 0.64 (dark).
- **Opening hook:** Ch 4's estimator assumes the host is in the catalogue; for 95% of
  events it is not (76 of 1588).
- **Sections:** 1. How incomplete? (C1). 2. The two-branch mixture (C2; the master
  equation unlocked in the hero graphic; w_G described at rung level — "the probability
  the host is catalogued", β_G/D mechanics deferred to Ch 9 per rung-guard).
  3. The completion numerator and the sky (C3; 5000× + sinθ; deviation D5 stated honestly).
  4. The zero-host fallback (C4; two-part story #54/#55 + biased-high #57). 5. The
  splinter (the Two Runaways planted; "hold that thought for six chapters").
- **Sources:** C1 — `G2c` D2 (GMV 2022 arXiv:2111.04629 Eq. 5; Gray 2023 arXiv:2308.02281
  Eq. 2.3); `pixel_completeness.py`; G7 row 15. C2 — `G2c` §1 + rows β_G/β_Ḡ/w_G + D4;
  Gray et al. 2020 Eqs. (9), (A14), (A15), (A19); `bayesian_statistics.py:3309-3311`;
  identity verification `HANDOFF_20260730.md` §2. C3 — `G2a_completion_sky_marginal_4pi.md`
  §3, §5, §7; `bayesian_statistics.py:3210-3238`; fixes `cb16142` + `4a259b7`; G7 rows 3–4;
  ledger #46. C4 — `H0R:1368-1420`; `8db6c6e`; ledger #54, #55, #57;
  `bayesian_statistics.py:2832-2844`. Splinter — measured class argmaxes from
  `real_r1/diagnostics/event_likelihoods.csv` (pedagogy's fresh measurement: in-cat 0.86 /
  dark 0.64).
- **Running example:** 889 is one of only 76; **dark event 606 introduced** (SNR 43,
  d_L 1.17 Gpc) as the permanent counterpart; both get dossier rows.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I5.1 The Two Branches | completeness fraction; per-branch display toggle | gen (w_G(h) 0.1625 → 0.0947 + branch leg sums from diagnostics CSV) | w_G has a *slope*, and the slope is a tilt independent of either branch's shape | static: w_G(h) curve + branch curves |
  | I5.2 The Two Runaways (plant) | class filter: in-cat / dark / both | gen (class-summed log posteriors, r1 + CRB `in_catalog`) | the headline number is a *crossing* — remember this | static: three curves peaking 0.86 / 0.64 / ~0.73 |
  | I5.3 The 4π Marginal | peak-evaluate vs marginalize toggle | js (toy sky-density integrand, chipped) + rec (the 5000×, G2a/ledger #46) | evaluating ≠ integrating, by 3.7 orders of magnitude | static: log-axis bar pair |
- **Questions:** Q5.1–Q5.5 (+ Traps 5.A, 5.B) — `BOOK_PEDAGOGY.md` §Ch 5.
- **Bias rail:** full-volume estimator at 0.0; a second amber pip appears labelled "two
  branches disagree (0.86 / 0.64) — unresolved" (no bias number; honesty > tidiness).
- **Museum interlude (end of page): "Information starvation"** — ledger #41 → #52: a
  formally-concluded fundamental limit, overturned as "a property of prior-INCONSISTENT
  estimators, not of the data". Links `museum.html#ex-starvation`.

---

### Ch 6 — Opening the Black Box: What the Waveform Actually Measures
`ch06-black-box.html` · `gen_ch06.py` · ~2,500 words

- **Discovery statement:** "an ellipsoid, somehow" is no longer good enough — how big, and
  correlated with what?
- **Learning goals:** noise-weighted inner product, PSD (+confusion noise), Fisher/CRB,
  fraction coordinates (φ, θ, u = d_L/d̂_L; +M_z/M̂_z later); why the GW leg is
  h-independent; two catastrophic-factor stories (dt²: SNR/10 → z ≤ 0.11 universe; frame:
  0.860 → 0.730 alone).
- **Opening hook:** the ball holds tens of thousands of galaxies — trust in either branch
  requires knowing the ellipsoid's size *and* orientation.
- **Sections:** 1. The inner product and the PSD (E1 formalized, E2). 2. The dt² story
  (G8; L1–L5 evidence lines; "a normalization that multiplies a threshold changes which
  data exists"). 3. Fisher and Cramér–Rao (E3; stencil history; singular-Fisher gate).
  4. Fraction coordinates (E4; Bishop 2.81–2.82 conditional marginalization; h-independence
  of the GW leg). 5. The frame story (ledger #12 fixed vs #27 non-cause — venue-scoped,
  do-not-rotate).
- **Sources:** E2 — `G8_dt2_inner_product_derivation.md`; `fcc49c4`, `49251f3`; G7 row 1;
  ledger #51; `LISA_configuration.py:_confusion_noise` (Babak et al. 2023 arXiv:2303.15929
  Eq. 17); Phase 9 / ledger #1. E3 — `parameter_estimation.py:399, :430`; Vallisneri 2008
  arXiv:gr-qc/0703086; Phase 10 / ledger #2; `d17230d` / G7 row 11 / ledger #11; PE-02 /
  ledger #13. E4 — `dark_siren_likelihood.md` §2.1, §7.3, §9.1; `G2a` §1;
  `bayesian_statistics.py:1856, :3532, :4014`. Frame — ledger #12, #27 (both, with the
  venue-scope rule).
- **Running example:** 889's real CRB row rendered as its sky ellipse + distance profile;
  the dossier gains "σ_dL/dL = 8.0×10⁻⁵, correlated with sky" and the Q6.5 comparison
  (photo-z error ≈ 6000× the distance error) sets up Ch 7.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I6.1 Fisher Ellipse Forge | SNR, sky position, "sky ⟂ distance" toggle | gen (real CRB rows) + js (analytic SNR rescaling, chipped) | the covariance is a *selection rule* — the toggle changes which galaxies are candidates | static: two ellipses, candidate counts annotated |
  | I6.2 The dt² Switch | one binary switch | gen (detected-population z-histograms with SNR vs SNR/10 at threshold 20, from the injection pool — h-invariant horizon trick) | a Riemann-sum factor is a different universe (z ≲ 1.5 → z ≤ 0.11) | static: the two z-histograms |
- **Questions:** Q6.1–Q6.5 (+ Traps 6.A, 6.B) — `BOOK_PEDAGOGY.md` §Ch 6.
- **Bias rail:** unchanged from Ch 5 (this chapter refines the instrument, not the estimator).
- **Museum interlude:** none (the Gaussian 3D/4D index non-bug #4 is annex-only).

---

### Ch 7 — A Redshift Is Not a Number
`ch07-redshift.html` · `gen_ch07.py` · ~3,000 words + interlude ~500

- **Discovery statement:** catalogue redshifts are photometric at σ_z/z ≈ 49%, and
  plugging in the central value *moves* the answer.
- **Learning goals:** likelihood ≠ posterior; Eddington-in-z; the σ_z² law
  (Δh = −Cσ_z², C_meas ≈ 17–20; coverage 0% → nominal); the `volume_deconv` kernel +
  Z_g + counted-once-in-z; PV/photo-z widths (K4); the point kernel's licence (K5);
  **the twist told in the fix's own chapter**: C7's measured inflation law and rail
  threshold σ_z/z > 0.256, presented per the G2b↔C7 conflict rule.
- **Opening hook (cold open):** drag σ_z up and watch the peak *slide*, not widen.
- **Sections:** 1. What a catalogue redshift is (K1; spec-z is 0.56% of GLADE+).
  2. The Eddington shift (K2; the four measured points; the coverage collapse).
  3. The volume kernel (K3; G2b RATIFIED; "dV_c counted once"; Z_g ∝ h⁻³ to 1e-15).
  4. Real widths (K4; PV 150/500 km/s; the 2.3×/4.9× dominance measurements; the
  citation-laundering anti-anchor as a cautionary aside). 5. The licence of the point
  kernel (K5; "generator-exact for the mock and wrong for real data"). 6. The twist (K6
  preview: the same knob over-corrects; the measured law + 0.256 threshold; **both G2b's
  ratified confirmation and C7's measurement shown side by side, conflict named, cell B
  named as decider** — sources map §7.1 verbatim requirement).
- **Sources:** K1 — `hostz_pv_photoz_kernel.md` §0–§1; `realistic_host_observation_model.md`
  §3.1. K2 — `G2b_host_z_volume_prior.md` §2.1–2.3 (RATIFIED), measured points
  `G2b:229-237`; G7 row 2; ledger #47. K3 — `G2b` §1, §3; `G2c` §4.3 D8;
  `bayesian_statistics.py:3712, :4202, :4190-4199`; ledger #75; gate `G2b:413-436`.
  K4 — `hostz_pv_photoz_kernel.md` (RATIFIED 2026-07-26; Laghi 2021 arXiv:2102.01708,
  Turski 2023 arXiv:2302.12037, Dálya 2022 arXiv:2110.06184, Davis 2011 arXiv:1012.2912);
  anti-anchor: GitHub #7 / `datamodels/galaxy.py:66`. K5 —
  `realistic_host_observation_model.md` §3.1–3.2 (RATIFY-R3);
  `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §4.3; ledger #88 (δ-kernel carries 85.3/86.7%).
  K6 — `CLAIM_2D_BIAS_20260730.md` C7 + `gate_b_20260730/C7_README.md` (the **corrected**
  law [1+√(1+12ε²)]/2 — never the claim's own 8ε² form); spec-z rescue refutation ledger #42.
- **Running example:** 889's host is photometric; its σ_z/z placed on the C7 law; the
  dossier's z slot finally fills — with an error bar wider than anything else on the card.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I7.1 The Eddington Machine | σ_z; kernel = bare / volume_deconv; "selection turnover" toggle | js (closed-form kernels/volume prior, chipped) + rec (the four measured Δh points + C law, `G2b:229-237`; the C7 law + 0.256 threshold, C7_README) | 3-stage: shifts *low* quadratically → deconv repairs it → the same knob over-corrects past 0.256 | static: 3 panels, one per stage, with the measured points |
  | I7.2 Spec-z Rescue Attempt | "keep only spec-z hosts" cut (predict-then-reveal) | rec (ledger #42: 0.56% sample, still rails 0.870) | the defect is the treatment of width, not the wide data | static: before/after posterior pair |
- **Questions:** Q7.1–Q7.6 (+ Traps 7.A, 7.B) — `BOOK_PEDAGOGY.md` §Ch 7.
- **Bias rail:** the σ_z ladder (−0.0016 … −0.046) livewired to I7.1's σ_z slider; repaired
  to −0.002 under volume_deconv; a **new amber pip: "C7: inflates at σ_z/z > 0.256 (live)"**.
- **Museum interlude (end of page): "volume_trunc"** — ledger #70 /
  `results/volume_trunc_ab_20260712/FINDING.md:1-58`: pre-registered fix, bias worse ~4×,
  two independent causes, `fixed_quad(n=50)` aliasing the GW peak to 0.0000. Lesson:
  quadrature is physics. Links `museum.html#ex-volume-trunc` (which carries the live
  quadrature dial M1).

---

### Ch 8 — A Second Handle: the Mass Channel
`ch08-mass-channel.html` · `gen_ch08.py` · ~3,200 words + interlude ~500

- **Discovery statement:** the mass channel should sharpen everything — and the bias goes
  to +0.077.
- **Learning goals:** M_z = M(1+z) at 10⁻⁴ as a second z-handle; the mass kernel family
  (production untruncated Gaussian, P(M<0)=4.8%; ratified truncated-lognormal CANDIDATE);
  G2d Eddington-in-M (with the sign-flip linearization lesson); M9 mechanism (z-broadening
  makes mz(z) a second z-likelihood — defects 2D-only); the amended mechanism
  (**de-weighting, not deletion**; 98.5% carried by survivors; the up-pull is the
  **(1−w_G) prefactor**, never "L_comp pulls up" — C10); C8 introduced (the
  reparametrization walk); the hard mass window's self-refutation (M6/HB).
- **Opening hook (the book's best cold open):** the reader was *promised* an improvement;
  10/10 runs pull +4.04σ.
- **Sections:** 1. The second handle (M1). 2. The kernel and its measure (M2; RATIFY-M6
  **CANDIDATE** badge mandatory at first mention). 3. Eddington in M (M3; the
  linearization sign-flip; the stale −0.020 comment flagged, re-measured −0.0022).
  4. Why only 2D breaks (M4; Eq. M9; [RATIFY-M7] caveats: catalogue masses are
  d_L-derived — hidden h-dependence up to δlnM ≈ 0.4). 5. What actually moved (the C3/C4
  amended budget: +15.83 = 0 + 19.10 − 3.27; w̄_cat 0.0354 → 0.0061; C10). 6. The
  well-posedness bomb (C8 introduced with the walk; adjudicated cause = the two numerator
  legs' measure mismatch, hard-wired to M_z,det,i). 7. HB, killed by its own author (M6).
- **Sources:** M1 — `dark_siren_likelihood.md` §7–§11; `G2c` D6;
  `bayesian_statistics.py:4014, :4363-4370, :3362`; Phase 15 / ledger #5; `f01595c` /
  ledger #22. M2 — `mass_marginal_2d_kernel.md` (RATIFIED gates M1–M7, **pairing = RATIFY-M6
  CANDIDATE**, `:1-17, 690-706`); Reines & Volonteri 2015 arXiv:1508.06274;
  exonerations ledger #72, #89 (venue-scoped, "necessary, not sufficient").
  M3 — `G2d_host_mass_rate_prior.md`; `bayesian_statistics.py:500`; stale comment
  `bayesian_statistics.py:2400-2401` vs ledger #50. M4 — `mass_marginal_2d_kernel.md` §3.5
  Eq. M9. C3/C4/C10 — `CLAIM_2D_BIAS_20260730.md` C3, C4 (amended), C10 +
  `gate_b_20260730/attack_c3_c4*.py`; the "84.2% is r1-specific" rule (sources map §7.5).
  C8 — `CLAIM_2D_BIAS_20260730.md` C8 + `gate_b_20260730/README_C8.md` +
  `c8_reparam.py`. M6 — `HANDOFF_20260730.md` §4 **as historical record only** + the claim
  file's exoneration (sources map §7.2).
- **Running example:** **I8.3 EMRI-889's Two Faces** — 889's clean 1D peak (0.745–0.750)
  vs nearly-flat 2D (0.79–0.80); dark 606's 80×-suppressed decreasing 2D leg vs rising
  completion. Dossier: "2D: destroyed this event's information and pushed it high."
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I8.1 The Impostor Sieve | σ_lnM slider (toy overlay) + 1D/2D channel switch (real) | gen (real culling stats: 64.7% zeros, 193:1 one-sidedness, w̄ 0.0354 → 0.0061 from diagnostics) + js (toy candidate set for the slider, chipped) | the sieve works *and* the weight collapses — only one was intended | static: rejection histogram + weight bar pair |
  | I8.2 The Reparametrization Walk | units dial (M☉ / 10⁵ M☉ / 10⁶ M☉ / kg) + "consistent everywhere" control | gen (reusing `gate_b_20260730/c8_reparam.py` outputs: MAP walk 0.81329 / 0.78107 / 0.74440 / rail 0.600; 1D bitwise-invariant; reconstruction gate 3.6e-12) | a published number that moves with an arbitrary unit is not a measurement | static: the four-MAP table + the invariant-1D row |
  | I8.3 EMRI-889's Two Faces | 1D/2D switch + event selector (889 / 606) | gen (term profiles of both events from diagnostics) | the whole class-level story is legible in two events | static: 2×2 panel grid |
- **Questions:** Q8.1–Q8.6 (+ Traps 8.A, 8.B) — `BOOK_PEDAGOGY.md` §Ch 8.
- **Bias rail:** "+0.077 (2D, 10/10 runs > +2σ)" appears in amber; the 1D row stays at
  its Ch 7 state. RATIFY-M6 candidate note on the rail tooltip.
- **Museum interlude (end of page): "mass_trunc"** — ledger #71→#72: confirmed in
  isolation (+0.016…+0.02, right sign), exonerated in the pipeline (Δ2D +0.0029 wrong
  sign; Δ1D 0.0000 exact) — *"the selection denominator is not a spectator."* Links
  `museum.html#ex-mass-trunc`.

---

### Ch 9 — Building a Universe to Break Your Estimator
`ch09-universe-factory.html` · `gen_ch09.py` · ~2,800 words

- **Discovery statement:** every claim so far compared to "truth" — where did truth come
  from, and does it agree with the estimator?
- **Learning goals:** the forward model (Barausse M1; Ω_m as design choice; plunge-window
  convention; injection pool); the realistic host-observation model (R2 forward
  convention: catalogue = TRUTH; R3 realized noise ≡ kernel width; R4 per-row widths, all
  photometric; R5 guards as executable derivations); **the two estimands** (N1–N4:
  Option A, the three legacy modes + de-rail matrix, `absolute_marginal` = #53,
  `generator_marginal` = #51); the Option-A identity failure on the real catalogue (N5:
  −17.2% / 33%); **C9** (w_G 2.3–2.5× off, z = −11.86, localized to mass-blind β_G).
- **Opening hook:** "biased relative to *what*?" — the mock recovered truth; the same
  code's generator and estimator disagree about the same universe by a factor 2.3.
- **Sections:** 1. The universe factory (population, plunge window, injections).
  2. Making the mock honest (R2–R5; the (A)-not-(B) argument; the honest price — declared
  truth = GLADE+'s observed distribution; the z-clip caveat, sources map §7.8).
  3. What am I normalizing to? (N1–N2; de-rail matrix 0.86 → 0.60 → 0.73 → 0.73; the
  Gray-literal citation correction C1). 4. Two estimands (N3–N4; w_G 0.1215 vs 0.0497 —
  the phantom 45% discrepancy; the disputed generator_marginal curve flagged, §7.7).
  5. The identity that fails (N5; G1). 6. C9 (measured; the adjudicator's leverage
  discount carried; the re-litigation guard — the exonerated *fix form* #61 must not be
  re-tried).
- **Sources:** forward model — Barausse 2012 arXiv:1201.5888; G11 / ledger #53;
  `plunge_window_initial_conditions.md`. R2–R5 — `realistic_host_observation_model.md`
  §1.2 (RATIFY-R1), §2 (RATIFY-R2), §3.4/§9 (RATIFY-R3/R9), §4 (RATIFY-R4 option c);
  `bayesian_statistics.py:112, :186, :273`; `handler.py:463`; P6 falsified-expectation
  (all 164 hosts photometric); §7.12 caveat (the false "host-miss rate logged" line —
  do not cite it). N1–N2 — `G2c` §3, §4.1–4.3 + §6 C1;
  `docs/gates/G3_ablation_cube.json`; ledger #49, #49a preview. N3 —
  `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §3.1; ledger #74, #77. N4 —
  `DERIVATION_GENERATOR_CONSISTENT_NORM.md` (full packet); ledger #81. N5 —
  `G1_beta_g_check.md:14-29`; `H0R:1548-1552`; the exonerated h⁻³ Jacobian (ledger:
  Option-A residual 1D-only +0.017). N6/C9 — `CLAIM_2D_BIAS_20260730.md` C9 +
  adjudication §5 + `c9_darkdraw_check.py`; KS D = 0.0863 dark-side extension; guard
  ledger #61.
- **Running example:** 889 under both estimands — same event, two w_G values; the reader
  sees "the probability 889's host is catalogued" is itself estimand-dependent.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I9.1 The Universe Factory | population params, plunge convention, mission span | js (toy population, chipped) + gen (real injection-pool marginals for the "real" preset) | the generator's conventions are estimator assumptions in disguise | static: (z, M, SNR) marginals, real preset |
  | I9.2 The Consistency Bench | generator mass-awareness dial × estimator w_G mass-awareness dial; live binomial z | gen (modeled w_G(h) both modes + realized 164/3135; mass-aware counterfactual 0.05149 / z = +0.21; **sig0_control used ONLY with its estimand caveat**, §7.6) | the shipped config reads z = −11.86, and the only dial that fixes it is the one nobody turned | static: w_G curves vs realized rate, z annotated |
  | I9.3 The De-rail Matrix | 4-step mode walker (pre-4π → 4π-only → local_ratio → volume_deconv) | rec (ledger #49: 0.86 → 0.60 → 0.73 → 0.73) | each fix earns its place; two of them were *necessary, not sufficient* | static: the four-MAP strip |
- **Questions:** Q9.1–Q9.5 (+ Traps 9.A, 9.B) — `BOOK_PEDAGOGY.md` §Ch 9.
- **Bias rail:** unchanged rows + a new amber pip "w_G ≠ realized rate (z = −11.86) — C9".
- **Museum interlude:** none in-page (absolute_marginal relocation #77 and the w_G
  bookkeeping relocation #61 are annex exhibits, linked from §4/§6).

---

### Ch 10 — Is It Calibrated?
`ch10-calibration.html` · `gen_ch10.py` · ~2,400 words + interlude ~400

- **Discovery statement:** it recovers truth at −0.24σ. That is not the same as being right.
- **Learning goals:** coverage, P–P plots, pulls over many universes (the independent
  harness — pure numpy, deliberately no production imports); what the closure test
  actually tested (133 golden events ≈ 100% of curvature → host association, not
  selection); the idealization ledger (0.032 km/s/Mpc is a consistency baseline, not a
  forecast; counterfactuals 8×–110×); σ→0 byte-identity gates (and why the σ→0 control is
  NOT the C6 control); pre-registration as an instrument; C11 (quantitative exoneration:
  6–16× too small — and the harness's own 1D-only scope).
- **Opening hook (predict-then-reveal):** "this estimator recovered truth at −0.24σ. Is it
  calibrated? y/n" — nearly everyone says yes.
- **Sections:** 1. Coverage is the currency (V1). 2. The instruments (V2 byte-identity;
  V3 closure at h=0.67 + redteam R2/R3; V4 pre-registration — "this book's
  predict-then-reveal is the same instrument"). 3. What the mock cannot see (R1
  idealization ledger; the honest sentence reproduced verbatim). 4. C11 — an exoneration
  done right (X4; the 1D-only scope caveat). 5. Failure planted: #51 → #53 changed three
  things at once.
- **Sources:** V1 — `validation/pp_coverage.py` module docstring; bare-kernel ≈0%
  coverage. V2 — `realistic_host_observation_model.md` §6.2 (RATIFY-R6); P5 md5s
  `1e81ba22`/`733c8d32`; Gate A1 caveat. V3 — ledger #20-era h=0.65, h=0.67 closure
  (MAP 0.670, +0.12σ); redteam ledger #85 (R2, R3); open thread #7 (sealed-truth mock —
  ordered, never run). V4 — `PREREGISTRATION_2x2_cellB.md`; P1–P6 scorecard,
  `REALISTIC_READOUT.md` (**with the §7.3 rule: the book follows C5, not the readout's
  "1D is defensible" sentence**). V5 — `G3_ablation_cube.json`;
  `G6_starvation_postmortem.md:12-16`. V6 — `threeway_ab/THREEWAY_AB_READOUT.md:19-56`;
  `mass_ab_20260727/MASS_KERNEL_AB_READOUT.md:23-80`. R1 —
  `idealization_audit/IDEALIZATION_LEDGER.md` I1–I4; `IDEALIZED_BASELINE_READOUT.md`
  (incl. `:50-52` median σ_z/z = 49%). X4/C11 — `CLAIM_2D_BIAS_20260730.md` C11;
  `results/pp_coverage_*/SUMMARY.md`.
- **Running example:** 889 is one of the 133 golden events carrying the closure — remove
  the top-K in I10.2 and the constraint (and 889's role in it) evaporates.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I10.1 The P–P Slot Machine | completeness fraction, σ_z; "run 200 universes" | gen (precomputed grid from `results/pp_coverage_*` outputs) | bias small while coverage collapses; coverage degrading with N (0.63 → 0.38 → 0.12) = real asymptotic bias | static: P–P curves at 3 settings |
  | I10.2 What Did the Closure Test Actually Test? | remove top-K golden events | gen (idealized combined posterior vs K, from **`run_seed61000/posteriors_fixed`** — canonical dir, see §5 data rules) | a passing test can pass for a reason you did not intend | static: constraint width vs K |
- **Questions:** Q10.1–Q10.5 (+ Traps 10.A, 10.B) — `BOOK_PEDAGOGY.md` §Ch 10.
- **Bias rail:** unchanged; the rail's tooltip now cites coverage numbers, not just MAPs.
- **Museum interlude (end of page): "The H₀-independent estimator"** — ledger #49a: MAP
  0.86 for *every* injected truth 0.63–0.77 while `catalog_only` tracked truth. Lesson:
  an estimator can be precise, stable, reproducible, and carry zero information. Links
  `museum.html#ex-h0-independent`.

---

### Ch 11 — The State of the Art, Honestly  *(closing chapter — must not resolve)*
`ch11-honest-state.html` · `gen_ch11.py` · ~3,000 words

- **Discovery statement:** three measured inconsistencies, a confounded attribution, and
  the control still running.
- **Learning goals:** reading an adjudication; C5 with its **binding fair-framing
  amendment** (both halves in the same paragraph, always); C7 as the G2b collision
  (present both, name cell B); C8 (well-posedness, unit-walk); C9 (+ dark-side KS
  extension); C6 (the three-variable confound; the dated pre-readout prediction
  reproduced so the reader sees pre-registration from the inside); the direction-not-verdict
  ending; the leverage payoff of Ch 5's splinter (±1/√N moves MAP up to 0.025;
  dh*/dε 1500–2400× idealized).
- **Opening hook:** *failure: everything. resolution: none yet — and that is the chapter.*
- **Sections:** 1. The scoreboard (C1–C11 status board). 2. C5, fairly (X1; concavity /
  extrapolated vertices 0.93–1.05 / interior peaks at extended grid; per-event cosmetic
  **and** class-summed +3.4–6.1σ). 3. The two runaways, unlocked (X1 crossing + leverage).
  4. Three inconsistencies, one side (K6/C7 with the G2b conflict; M5/C8; N6/C9; C10
  prefactor rule; X3). 5. The confound and the control (X5/R6; cell B pre-readout
  prediction, dated). 6. Open threads (X6, incl. the foreground-impostor *hypothesis* —
  labelled hypothesis, not measurement; and thread 7, the sealed-truth mock never run).
  7. What would you do first? (the Q11.6 discipline).
- **Sources:** the current adjudicated state **only**:
  `CLAIM_2D_BIAS_20260730.md` (as amended 2026-07-30) +
  `gate_b_20260730/ADJUDICATION_20260730.md` + `BIAS_HISTORY_LEDGER.md`; X1–X6 cards in
  `BOOK_SOURCES_MAP.md` §3 Tier 10 (which carry the artifact lines); leverage numbers from
  the claim file's C5 block; C10 budget (+31.55 / −3.11, 39.1%); C11 from Ch 10. The
  nats→h conversion, if used anywhere, in the corrected form (sources map §7.20).
- **Running example (the payoff):** 889's channel swing +1.98 → −2.04 → −3.30 across
  r1/r2/r3, flipping its class's sign in r3 — the dossier's final row is a warning label,
  and the reader's Ch 3 host-guess marker is re-surfaced (`Book.getPrediction`).
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | I11.1 The Two Runaways, Unlocked | class reweight (±1/√N), λ-scan, idealized/realistic toggle | gen (class sums + precomputed reweight curves, r1 + idealized baseline) | your own hands move the headline number by 0.025 — and 0.0000 idealized | static: MAP-vs-ε curves, both venues |
  | I11.2 The Adjudication Board | filter C1–C11 by status / provenance tag / estimator side | gen (hand-authored claims JSON from the claim file + adjudication — statuses must match verbatim) | three measured inconsistencies on one side is a *direction*, and the board shows why not a verdict | static: the board as a table |
- **Questions:** Q11.1–Q11.6 (+ Traps 11.A, 11.B) — `BOOK_PEDAGOGY.md` §Ch 11. Q11.6 is
  the book's last question; the chapter ends on open threads with **no answer key**, per
  the sources map X6 hook.
- **Bias rail:** the full honest amber state — 1D at its contingent 0.0 (with the
  "near-cancellation" tooltip from C5), 2D at +0.077, three amber pips C7/C8/C9, one grey
  pip "C6: attribution confounded — cell B in flight".
- **Museum interlude:** none in-page; closes with a link to the annex as "the 98
  hypotheses this chapter is standing on".

---

### Museum annex — The Defect Museum
`museum.html` · `gen_museum.py` · ~2,500 words + exhibit data

- **Discovery statement:** 98 hypotheses, and what each one cost.
- **Structure:** the 12 exhibits of `BOOK_SOURCES_MAP.md` §4 (verbatim table as the
  spine), plus **#ex-h0-independent** (ledger #49a) and **#ex-archaeology** (#49b,
  timeline scrubber M2). Each exhibit: hypothesis · pre-registered prediction · decisive
  test · measured outcome · what it cost · transferable lesson. Anchor ids (FIXED — the
  five interlude-carrying chapters link them):
  `#ex-volume-trunc, #ex-mass-trunc, #ex-pdet-anchor, #ex-zerohost-fallback,
  #ex-starvation, #ex-gray-mixture, #ex-wg-bookkeeping, #ex-ha-measure, #ex-hb-window,
  #ex-absolute-marginal, #ex-numerator-only, #ex-index-bug, #ex-h0-independent,
  #ex-archaeology`.
- **Interactives:**
  | id | manipulate | src | AHA | fallback |
  |---|---|---|---|---|
  | M1 Falsified Fix Gallery (flagship: volume_trunc quadrature dial) | Gauss–Legendre node count n | js (genuine re-computation: fixed-order nodes vs a narrow GW-peak integrand — reproducible in-browser) + rec (the pipeline numbers, FINDING.md) | at n=50 the integral reads 0.0000 where the exact value is 0.24–0.65 — the peak falls between nodes | static: integrand + node positions at n=50 vs n=400 |
  | M2 Archaeology | timeline scrubber over stored-posterior eras | gen/rec (dates + MAPs from ledger #49b: unbiased 2026-04-09 → rail born ~2026-04-24) | "when did it break?" is a question with a method | static: MAP-vs-date strip |
- **Also owns:** `data/museum_ledger.json` — the 98-row ledger digest powering the
  phase-4 BW3 instrument (row #, hypothesis, verdict, venue, date, decisive artifact).
- **Museum meta-rule (binding):** any interactive anywhere that lets the reader "try" a
  hypothesis in the claim file's Exonerated list **plus** ledger §2 items 1–17 must reveal
  the measured verdict, never leave it open.

---

## 2. File-ownership map (the fan-out contract)

One agent per chapter. Each agent owns **exactly** these files and nothing else:

| Agent | Owns (create/edit) |
|---|---|
| ch00 | `book/site/ch00-two-numbers.html`, `book/generators/gen_ch00*.py` (optional), `book/site/data/ch00_*.json` |
| ch01 | `book/site/ch01-ruler.html`, `book/generators/gen_ch01*.py`, `book/site/data/ch01_*.json` |
| ch02 | `book/site/ch02-bayes.html`, `book/generators/gen_ch02*.py`, `book/site/data/ch02_*.json` |
| ch03 | `book/site/ch03-which-galaxy.html`, `book/generators/gen_ch03*.py`, `book/site/data/ch03_*.json` |
| ch04 | `book/site/ch04-loud-half.html`, `book/generators/gen_ch04*.py`, `book/site/data/ch04_*.json` |
| ch05 | `book/site/ch05-unseen-galaxy.html`, `book/generators/gen_ch05*.py`, `book/site/data/ch05_*.json` |
| ch06 | `book/site/ch06-black-box.html`, `book/generators/gen_ch06*.py`, `book/site/data/ch06_*.json` |
| ch07 | `book/site/ch07-redshift.html`, `book/generators/gen_ch07*.py`, `book/site/data/ch07_*.json` |
| ch08 | `book/site/ch08-mass-channel.html`, `book/generators/gen_ch08*.py`, `book/site/data/ch08_*.json` |
| ch09 | `book/site/ch09-universe-factory.html`, `book/generators/gen_ch09*.py`, `book/site/data/ch09_*.json` |
| ch10 | `book/site/ch10-calibration.html`, `book/generators/gen_ch10*.py`, `book/site/data/ch10_*.json` |
| ch11 | `book/site/ch11-honest-state.html`, `book/generators/gen_ch11*.py`, `book/site/data/ch11_*.json` |
| museum | `book/site/museum.html`, `book/generators/gen_museum*.py`, `book/site/data/museum_*.json` |
| **integrator** (last) | ALL shared files below + retiring the legacy demo (`ch00-demo.html`, `gen_ch00_demo.py`, `ch00_demo.json`) + flipping manifest statuses to "live" + WIDGET_REQUESTS triage + BW2/BW3/persona-switch instruments |

**FROZEN as of this document** (integrator-only; capability requests go to
`book/design/WIDGET_REQUESTS.md`, workarounds go inline in the requesting chapter's own
HTML file):

```
book/site/css/book.css          book/site/js/book.js        book/site/js/manifest.js
book/site/_template.html        book/site/index.html        book/generators/make_all.py
book/site/vendor/**             .github/workflows/ci.yml    book/README.md
book/design/BOOK_DESIGN.md      book/design/BOOK_SOURCES_MAP.md
book/design/BOOK_PEDAGOGY.md    book/design/BOOK_TECH_DESIGN.md
```

(`WIDGET_REQUESTS.md` is append-only for chapter agents — append request blocks below the
marked line; never edit existing blocks.)

Mechanics already in place (built/fixed this phase — chapter agents rely on, never edit):
- `make_all.py` **auto-discovers** `gen_ch*.py` / `gen_museum*.py`: dropping your
  generator in is registration.
- The **nav** is built from `js/manifest.js` (`Book.buildNav`) — pages carry
  `<nav class="book-nav" data-nav>` and never hardcode chapter links. The integrator flips
  your entry to `live`.
- `_template.html` is the required page skeleton (topbar, voices, gw-reader stratum,
  dossier, prov-chips, badges, self-check, provenance panel, bias rail, script order).
- `book.js` provides: `themedPlot` (theme-reactive Plotly), `gridSlider`,
  `predictReveal` (+ localStorage persistence via `data-predict-id`, `getPrediction`),
  `combineLogRows` / `normalizePosterior` / `trapz` / `argmaxIdx` / `logsumexp`,
  `biasRail`, `loadJSON`, `renderMath`, `isDark`.
- CSS classes: `prov-chip[ real|toy]`, `badge (ratified|candidate|finding|open|refuted|
  exonerated|confounded)`, `voice-derivation`, `voice-adjudicator`, `gw-reader`,
  `dossier`, `num-table`/`num-view`/`table-scroll`, `toggle-row`, `provenance-panel`,
  `callout (without|defect)`, `widget`, `selfcheck`/`answer`.

**Cross-chapter data duplication is accepted by design.** I5.2 and I11.1 both need
class-summed posteriors: each chapter's generator computes its own copy into its own
`data/chNN_*.json`. Zero coupling beats deduplication here; generators are cheap and
deterministic.

---

## 3. The consistency contract

### 3.1 Notation table (binding across all chapters)

Symbols exactly as the project writes them. Chapter agents mark each occurrence
`<span class="term" data-term="KEY">…</span>` (inert until the phase-4 Symbol Passport).

| KEY | Symbol | Meaning | Units | Defining source |
|---|---|---|---|---|
| h | $h$ | $H_0/100\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$; mock truth $h_{\rm true}=0.73$ | — | `constants.py` H |
| H0 | $H_0$ | Hubble constant | km s⁻¹ Mpc⁻¹ | `constants.py` |
| dL | $d_L$ | luminosity distance | Mpc (Gpc for pools) | `physical_relations.py:132 dist` |
| z | $z$ | true redshift | — | `dark_siren_likelihood.md` §2.4 |
| zg | $z_g$ | catalogue (observed) redshift of galaxy g | — | `handler.py`; K1 |
| sigz | $\sigma_z$ | host-z kernel width (total: meas ⊕ PV) | — | `hostz_pv_photoz_kernel.md` |
| eps | $\sigma_z/z$ | fractional z width (C7 variable, rail at 0.256) | — | C7_README |
| Om | $\Omega_m$ | matter density; fiducial 0.2726 (Barausse M1, design choice) | — | `constants.py`; G11 |
| Ez | $E(z)$ | $\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}$ | — | `dark_siren_likelihood.md` §2.4 |
| dVc | $dV_c/dz$ | comoving volume element per unit z (per sr where noted) | Mpc³ | `physical_relations.py:571` |
| wpop | $w_{\rm pop}$ | $(dV_c/dz)/(1+z)$ — the volume/rate prior in z | Mpc³ | `G2b` §1 |
| Zg | $Z_g$ | per-galaxy kernel normalization ($\propto h^{-3}$, exact) | Mpc³ | `G2b` §3 |
| M | $M$ | MBH (source-frame) mass | M☉ | `datamodels/parameter_space.py` |
| mu | $\mu$ | compact-object mass | M☉ | `parameter_space.py` |
| Mz | $M_z$ | redshifted (detector-frame) mass $M(1+z)$ | M☉ | `dark_siren_likelihood.md` §7 |
| Mg | $M_g$ | catalogue BH-mass proxy of galaxy g (d_L-derived — hidden h-dependence, RATIFY-M7) | M☉ | `mass_marginal_2d_kernel.md` |
| slnM | $\sigma_{\ln M}$ | mass-proxy scatter (≈0.58 kernel / ≈1.28 catalogue-side; state which) | — | `mass_marginal_2d_kernel.md`; RV15 |
| snr | SNR / $\rho$ | matched-filter signal-to-noise; threshold 20 | — | `parameter_estimation.py:488`; `constants.py` |
| Gam | $\Gamma_{ab}$ | Fisher matrix $\langle\partial_a h\|\partial_b h\rangle$ | mixed | `parameter_estimation.py:399` |
| Sig | $\Sigma$ | CRB covariance $\Gamma^{-1}$ | mixed | `parameter_estimation.py:430` |
| u | $u$ | fractional distance $d_L/\hat d_L$ (mean 1) | — | E4; `bayesian_statistics.py:1856` |
| phth | $\phi,\theta$ | sky coordinates (frame-stamped) | rad | `LISA_configuration.py` |
| wg | $w_g$ | rate weight $R_{\rm eff}(M_g)/(1+z_g)$ | yr⁻¹-ish (relative) | `bayesian_statistics.py:879`; G2c D1 |
| Reff | $R_{\rm eff}(M)$ | per-MBH EMRI rate | yr⁻¹ | Babak 2017 arXiv:1703.09722 |
| Ng | $N_g$ | per-galaxy numerator integral | — | `G2c` §2 |
| Dg | $D_g$ | per-galaxy selection integral | — | `G2c` §2 |
| Lcat | $\mathcal L^{\rm cat}$ | catalogue leg $\sum w_g N_g/\sum w_g D_g$ (ratio of sums) | — | `G2c` §2, §4; ledger #26 |
| Lcomp | $\mathcal L^{\rm comp}$ | completion leg $B^{\rm num}/\beta_{\bar G}$ (diagnostic identity D4) | — | `G2c` D4 |
| Bnum | $B^{\rm num}$ | completion numerator (1/4π sky marginal) | Mpc³ sr⁻¹ | `G2a`; `bayesian_statistics.py:3210-3238` |
| betaG | $\beta_G$ | catalogued-side selection integral | Mpc³ sr⁻¹ | `G2c`; `bayesian_statistics.py` (β_G = D − β_Ḡ) |
| betaGbar | $\beta_{\bar G}$ | dark-side selection integral | Mpc³ sr⁻¹ | `bayesian_statistics.py:1170` |
| Dh | $D(h)$ | full-volume selection normalization $\beta_G+\beta_{\bar G}$ | Mpc³ sr⁻¹ | `bayesian_statistics.py:1013`; G2c §6 C2 (cite "denominator of (A14)") |
| wG | $w_G$ | mixture weight $\beta_G/D$ — **estimand-dependent**; always name the mode | — | `bayesian_statistics.py:3309-3311`; C9 |
| pdet | $p_{\rm det}$ | detection probability (horizon-survival estimator) | — | `simulation_detection_probability.py` |
| fz | $f(z,\Omega)$ | catalogue completeness fraction | — | `G2c` D2; `pixel_completeness.py` |
| pi | $p_i(h)$ | per-event likelihood (the master equation) | — | `bayesian_statistics.py:3006-3009, 1042-1048` |
| Cscale | $C$ | the arbitrary mass-coordinate rescale of the C8 walk | — | `README_C8.md` |
| sigh | $\sigma_h$ | posterior width in h | — | readouts |

Master equation, hero graphic and per-chapter unlock state: exactly as in
`BOOK_PEDAGOGY.md` §2.0 (factors greyed until their chapter; three ambered by Ch 11).

### 3.2 Citation style

- **Internal artifacts:** inline `<span class="prov-chip">G2b §2.1–2.3</span>` /
  `<span class="prov-chip">bayesian_statistics.py:3309-3311</span>` /
  `<span class="prov-chip">ledger #49a</span>`. Line numbers are re-grep anchors, not
  immutable — copy them from `BOOK_SOURCES_MAP.md`, do not invent.
- **External literature:** `Author et al. (year), arXiv:ID` — e.g. Gray et al. 2020,
  arXiv:1908.06050. No bibliography page; citations are inline.
- **Status badges:** the project's tiers verbatim (`RATIFIED / CANDIDATE / FINDING /
  OPEN / REFUTED / EXONERATED / CONFOUNDED`) via `.badge` classes, at **first mention** of
  the object they qualify, per the trust-tier licence table in `BOOK_SOURCES_MAP.md` §0.
- **The three voices** (pedagogy writing rule 7): narrator = plain prose; ratified
  derivations = `.voice-derivation`; adjudicated findings/amendments =
  `.voice-adjudicator`. Never blend a finding into narrator voice.
- **Every widget** carries `prov-chip real: <run>` or `prov-chip toy: analytic`
  (pedagogy interaction principle 3). Recorded-measurement widgets (`rec`) use
  `prov-chip real` + the artifact chip.

### 3.3 The fidelity rule (hard constraints)

1. **No equation without a source; no number without provenance.** Every equation cites
   its ratified packet or code site; every measured number carries its artifact chip. The
   book *cites*, never loosely re-derives (project constraint, non-negotiable).
2. **The C7/C8/C9 state must match the adjudication** —
   `CLAIM_2D_BIAS_20260730.md` (as amended 2026-07-30) + `ADJUDICATION_20260730.md`. All
   three are **live**: explained, never retired. C7 uses the **corrected** law
   $[1+\sqrt{1+12(\sigma_z/z)^2}]/2$ (never the claim's 8ε² form) and is presented
   **jointly with G2b's ratified confirmation** — both sides, conflict named, cell B named
   as decider (sources map §7.1). C8's cause is the numerator-leg measure mismatch (not
   "4D vs 3D D(h)"). C9 carries the adjudicator's leverage discount and the #61
   re-litigation guard.
3. **Binding amendment pairs** — never quote half: C5 (per-event cosmetic **and**
   class-summed +3.4–6.1σ); C4 (deletion refuted → de-weighting, 98.5% survivors); C10
   (the up-pull is the (1−w_G) prefactor; $\mathcal L^{\rm comp}$ pulls *down* for dark
   events — the sentence "the completion term pulls up" is banned); "84.2%" is
   r1-specific (print the replicated qualitative claim, not the number, as the finding).
4. **Venue-scope every negative result** (standing rule; #70/#72 share one 494-event
   subsample). RATIFY-M6: any 2D number is presented as sitting on **CANDIDATE** ground.
5. **The do-not-retry union is binding** (claim file Exonerated list + ledger §2 items
   1–17): sandboxes that reach a dead configuration must volunteer the verdict.
6. **The full conflict register `BOOK_SOURCES_MAP.md` §7 (items 1–20) is incorporated by
   reference** — including: HANDOFF §4 never cited as current (§7.2); the book follows C5
   over REALISTIC_READOUT §6 (§7.3); sig0_control only with its estimand caveat (§7.6);
   the disputed generator_marginal w_G curve (§7.7); the z-clamp caveat (§7.8); contested
   #37–#39 (§7.9); `DRAFT_REPORT.md` cited by its real name (§7.10); stale −0.020 comment
   (§7.11); the two wrong code-comment citations never propagated (§7.13); nats→h in the
   corrected form only (§7.20).
7. **Rung-guard** (pedagogy §1.3): no tool from a higher rung, even in an aside; use
   forward-reference chips ("⏭ Ch 6") instead.

### 3.4 Review rubric (phase-4 reviewers score each chapter 0–3 per axis)

**A. Student comprehension (Mara test)**
- 3: every new symbol has units + physical picture before its first equation; the chapter
  is followable without opening any linked derivation; the discovery arc is causal
  (failure → tool → measured repair), not expository.
- 2: minor rung violations or one unmotivated symbol. 1: relies on a linked doc to
  complete an argument. 0: presents formulas before failures.

**B. Physics fidelity (Examiner test)**
- 3: every number traced to its chipped artifact and correct as quoted; badges correct at
  first mention; all §3.3 hard constraints satisfied; voices separated; venue scoping
  present; the provenance panel is complete (spot-check ≥5 chips against artifacts).
- 2: chips complete but ≤2 badge/voice slips. 1: any unchipped measured number.
  **0 (automatic): any violation of §3.3 items 2, 3, or 5 — banned-sentence check
  included.**

**C. Interaction value**
- 3: each widget passes the pedagogy Part 4 gate — MANIPULATE/OBSERVE/AHA stated, the
  "lost-if-static" is real, real-data widgets chipped `real`, toys chipped `toy`;
  predict-locks enforced; every widget has a numbers view + static fallback; no
  differentiation across the h-grid seams; break-it states recoverable in one click with
  the fix's name.
- 2: one widget is decoration (weak lost-if-static). 1: toy presented as data or a
  missing provenance chip. 0: an interactive contradicts a measured verdict or leaves a
  dead hypothesis open.

**D. Question quality**
- 3: questions verbatim from `BOOK_PEDAGOGY.md` Part 3 (or improvements that keep the
  mechanism-naming answers + provenance); answers hidden; last question is the transfer
  whose answer is the next chapter's opening failure; traps in the reader's voice,
  dismantled with a measurement.
- 2: one answer names the outcome but not the mechanism. 1: recall questions present.
  0: answers visible by default or transfer question missing.

**Ship gate:** A ≥ 2, B = 3, C ≥ 2, D ≥ 2. B is non-negotiable.

---

## 4. Build order, parallelization, and prohibitions

### 4.1 Order

All 13 chapter/museum agents are **file-disjoint and may run fully parallel**. The
recommended 3-wave schedule optimizes review flow and tone-setting, not dependencies:

- **Wave 1 (tone-setters):** Ch 4, Ch 7 — the two flagship "without this it breaks"
  chapters; their voice calibrates everything else. Plus Ch 0, Ch 1, Ch 2 (lightest data).
- **Wave 2:** Ch 3, Ch 5, Ch 6, Ch 8, museum annex (the annex early — five chapters link
  its FIXED anchors, which are specified in §1, so no coordination is actually required).
- **Wave 3:** Ch 9, Ch 10, Ch 11 (the honesty-critical chapters; benefit from reviewed
  Wave-1/2 precedent).
- **Integrator (last):** manifest flips to `live`, legacy-demo retirement, index contents
  links, WIDGET_REQUESTS triage, BW2/BW3/persona instruments, cross-chapter link check,
  full-site serve smoke test, CI verification.

Per-agent workflow: generator first (verify JSON against the numbers quoted in §1 — if a
generator's recomputation disagrees with a spec number, **stop and flag; do not
silently reconcile in either direction**), then the page from `_template.html`, then
self-review against §3.4.

### 4.2 Data-hygiene rules for generator authors (binding; from sources map §5/§7.19)

1. Canonical posterior dirs: realistic runs
   `results/campaign51_20260728/realistic_20260729/seed{61000,62000}/real_r{1..5}/posteriors/`
   are canonical as-is; **idealized baselines**: seed61000 → `run_seed61000/posteriors_fixed`
   (plain `posteriors/` is the stale pre-`ec09ed0` backup), seed62000 →
   `run_seed62000/posteriors`.
2. **Never** compute on the root/idealized diagnostics CSVs without era disambiguation
   (they hold two concatenated evaluate sweeps; 2.00× row counts).
3. The h-grid is non-uniform (0.01 on [0.60,0.65] ∪ [0.80,0.86]; 0.005 on [0.65,0.80]).
   No second differences across the seams; `Book.trapz` handles non-uniform integration.
4. Never use the 4-dp `w_G` log line for residual-level work; derive from D(h)/β_Ḡ at
   full precision.
5. The local `reduced_galaxy_catalogue.csv` differs from the #53 parent **only in
   `z_error`** — usable for positions/z displays (I3.1), not for width-sensitive work.
6. `sig0_control` carries the `generator_marginal` estimand — use only with the caveat.
7. Generators: deterministic (fixed seeds), repo-root via
   `Path(__file__).resolve().parents[2]`, read-only outside `book/`, output compact JSON
   (target < 500 KB per file; slice/downsample server-side, not in the browser).

### 4.3 What chapter agents must NOT do

1. **Never write outside your owned files** (§2). Never edit frozen files — shared-widget
   needs go to `WIDGET_REQUESTS.md` (append-only) with a page-local workaround.
2. **Never write to `/home/jasper/Repositories/MasterThesisCode`** (the main worktree) —
   generators read it; nothing writes it. Never edit `master_thesis_code/` in either tree.
3. **No re-derivation.** If a step seems to need one, the chapter is over-reaching — cite
   the packet or cut the step.
4. **No invented, adjusted, or "rounded-up" numbers.** Every number comes from
   `BOOK_SOURCES_MAP.md`, `BOOK_PEDAGOGY.md`, or the cited artifact, carried verbatim.
   Generator-recomputed values that disagree with spec numbers are flagged, not shipped.
5. **No banned sentences:** "the completion term pulls up" (C10); "impostor deletion is
   the mechanism" (C4-amended); "the volume kernel is calibrated, full stop" / "the kernel
   is wrong, settled" (G2b↔C7); "the 1D channel is fine" (Trap 11.B); "58% of hosts rail"
   without the fair-framing amendment; "84%" as the replicated finding.
6. **No external network references** (CDN scripts, fonts, images — the site is
   self-contained; CSP-equivalent by policy), no absolute filesystem paths in any shipped
   file, no new JS/Python dependencies, no build tooling.
7. **No un-attributed toys.** A toy that looks like data is a scientific-integrity
   failure in a book about scientific integrity.
8. **Do not resolve Ch 11.** The leg question ends as "a direction, not a verdict"; open
   threads end open; the last self-check block has no answer key where the project has
   none.
9. **Do not touch git state** beyond committing your own files on the current branch
   (`book/foundations-interactive`); no pushes, no PRs, no rebases — the integrator ships.

---

*End of build spec. GO for fan-out per §4.1.*
