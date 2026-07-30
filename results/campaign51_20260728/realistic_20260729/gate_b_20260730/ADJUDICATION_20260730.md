# ADJUDICATION — Gate B/C, campaign #53 2D-bias claim set (2026-07-30)

Adjudicator: fable/xhigh, per `RUNBOOK_NEXT_SESSION_6.md` §4. Inputs: the five
Gate B/C attacker reports, `CLAIM_2D_BIAS_20260730.md` (the claim under attack),
`HANDOFF_20260730.md`, `PREREGISTRATION_2x2_cellB.md`, and the Gate A results
(A1–A3, all [LOCAL, VERIFIED]).

**Adjudicator's own verification (not taken on trust):** re-ran
`attack_c3_c4.py` — C3 split +2.97 / +15.83 / +18.80 (84.2% dark) and every C4
observation reproduce exactly. Checked `c4_decomposition_results.json` closes
internally (dark: −t1 + t2 = 19.10 − 3.27 = 15.83; partition 0.24 + 0.00 +
15.60 = 15.83; prefactor bookkeeping 28.45 = 31.55 − 3.11, in-cat 21.12 =
1.51 + 19.61, dark 7.33 = 30.04 − 22.72 — all consistent). Checked
`c8_reparam_results.json` (C-walk, 1D bitwise flag), `c8_canonical_measure_results.json`
(0.849171 = HA's exonerated 0.8492 to 3e-5), `g4_results.json` (counterfactual
2D 0.8123→0.7433, 1D 0.7321→0.6430), `g6_results.json`, `c7_orient_results.json`
(L_cat argmax 0.86 for 66/74 with 2 NaN; combined 44/76 — the mixture-weight
h-dependence resolves the apparent tension), and the C4 class profiles
(1D in-cat rise +3.92, 2D +7.44, completion +33.09, dark completion argmax
0.810 — all match the reports). No discrepancies found.

Verdicts use: **FINDING** (promoted), **REFUTED**, **AMENDED** (core survives,
stated mechanism/framing does not), **UNDETERMINED**. Per the runbook, refuted
and undetermined are valued outputs; none of the verdicts below is forced.

---

## 1. Per-claim adjudication

### C1 — 1D class budget → **FINDING** [LOCAL, VERIFIED, replicated]
Gate A3: replicates in sign and order across all 10 realistic runs and both
seeds (in-cat +1.27..+5.38 vs idealized −338/−248; dark −11.8..−14.1). The
claim's own refutation route was executed and failed to refute. Closed.

### C2 — channel totals → **FINDING** [LOCAL, VERIFIED]
Direct read of delivered posteriors; independently reconstructed by the C8
attacker to 3.6e-12 nats and cross-checked (1D −9.30, 2D +9.51). Closed.

### C3 — 84% of the channel difference is the dark class → **promoted to FINDING** [LOCAL, VERIFIED]
- Strongest evidence for: Gate A2 reproduced the split **exactly** from re-pulled
  cluster data (+2.97 / +15.83, dark share 84.2%), and the adjudicator re-ran the
  script this session. The provenance defect that made this the weakest number is
  cured: the diagnostics CSV is re-staged locally.
- Strongest residual weakness: the split is **r1-only** (the diagnostics CSV
  exists for no other run). C1's class structure replicates everywhere, so
  r1-specificity is unlikely, but the 2D partition itself is single-realization.
- Verdict: FINDING, with an explicit r1-only caveat until the diagnostics CSV is
  emitted on every run (instrumentation item, already routed).

### C4 — mechanism "impostor rejection → completion fallback" → **observations FINDING; mechanism AS STATED REFUTED; replaced by an AMENDED mechanism** [LOCAL, VERIFIED, r1]
- The **observations** (64.7% dark `L_cat_with_bh == 0`; 488/1095 2D-zero at
  every h, 487 dark; median suppression 7.78e-3; dark Σln(Lcat2D/Lcat1D) tilt
  −504.8) all reproduce → promote to [LOCAL, VERIFIED].
- The **stated mechanism is refuted on two exact grounds** (attacker 1, algebra
  verified to 6.2e-13 on all 65,108 cells):
  1. Writing p = C(1+R), C = (1−w_G)L_comp channel-common, ln C **cancels
     identically** from the per-event channel difference. "Falls back on the
     completion term, which pulls up" cannot appear in the +15.83 **as an
     accounting statement about the channel difference**.
  2. The flagship evidence (487 events 2D-zeroed at every h) carries **+0.24
     nats = 1.5%** of the +15.83. The 491 both-dead events carry exactly 0.00.
     **98.5% (+15.60) is carried by the 534 survivors.** Deletion is not the
     mechanism; the zeroed events' 1D legs were already negligible.
- **Amended mechanism (promoted):** the mass kernel **de-weights** the surviving
  dark catalogue legs (dark mean catalogue mixture weight 0.0354 → 0.0061 at
  h=0.73, a 5.8×). Exact budget: +15.83 = 0 (completion, cancels) + 19.10 (loss
  of the 1D catalogue down-tilt; +18.87 survivors, +0.24 zeroed) − 3.27
  (residual 2D tilt). The dark class-summed opposition over 0.73→0.86 collapses
  −24.46 → −0.63 nats and its argmax moves 0.640 → 0.785 — landing next to the
  dark **completion leg's own argmax 0.810 ≈ the delivered 2D MAP 0.8133**.
- **Adjudicator's caution against over-correcting:** the refutation is of the
  *accounting*, not of the completion leg's role in the **absolute** 2D
  position. Once the dark catalogue leg is de-weighted, the 2D posterior *does*
  sit where the channel-common completion/prefactor structure puts it. But the
  up-pull is **prefactor-carried**: dark Σdln[(1−w_G)L_comp] = +7.33 = +30.04
  from N·Δln(1−w_G) **minus** 22.72 from L_comp itself. **L_comp pulls DOWN for
  dark events** (only 39.1% positive tilts). "The completion term pulls up" must
  be rewritten as "the (1−w_G) prefactor pulls up" (new claim C10).
- Caveat: the entire partition is seed61000/real_r1 (only run with a diagnostics
  CSV).

### C5 — 58% of in-cat hosts rail at the prior edge → **FINDING, interpretation AMENDED** [LOCAL, VERIFIED, replicated]
- **The rail survives its designated refutation attempt (edge-artifact):**
  railed profiles are genuinely concave (86–96% all-negative second differences
  on the uniform 0.80–0.86 stretch, |d²| ~1e11× roundoff) and top-K parabola
  vertices give finite implied peaks h_eff = 0.93–1.05 (median), stable over
  K = 3–9, all 10 runs — with the extrapolator *validated in-band* (median error
  < 0.007 at the relevant standoffs). Independently, the C7 attacker rebuilt the
  single-host kernel on a grid extended to h = 2.4 and found **interior** peaks
  (median ≈ 1.12): the 0.86 concentration is a **clipped real runaway, not an
  edge artifact**. Replicates in 10/10 runs (54–67% at edge vs 2.4% flat-surface
  expectation, 5.3% idealized).
- **Fair-framing amendment (binding for any write-up):** per event the rail is
  cosmetic — median peak-to-truth Δln p 0.072–0.134 nats, i.e. 0.30–0.47 σ_event
  (implied per-event σ_h 0.235–0.311 vs 0.043–0.053 idealized); 0–1.3% of events
  exceed 1σ. "58% of hosts rail" invites over-reading. But it is **not noise**:
  the tilt is coherently same-signed, and the **class-summed** displacement is
  +3.4 to +6.1 σ_class above truth in 8/10 runs, two independent σ routes
  agreeing, LOO never moving the argmax. A correct estimator under large σ_z is
  wide but centred; a ≥3.4σ coherent class displacement is not that. **"Not a
  centred measurement" stands.**
- **Attribution amendment (two-component):** the claim's "the identified hosts
  … prefer the top of the prior" splits:
  * The **per-event argmax rail lives in the catalogue leg** (L_cat argmax at
    0.86 for 66/74 = 89.2%; L_cat carries a median 96.3% of the in-cat mixture
    at h = 0.73) — and **C7 is its confirmed mechanism** (below).
  * The **class-summed mixture rise** (+3.92 nats 1D over 0.73→0.86) is ~82%
    carried by the ~9%-weight **completion admixture** (catalogue leg class-sum
    peaks at 0.760 with rise only +0.80 — a few golden events' large negative
    tilts nearly cancel the many small positive ones; the per-event median
    +0.308 with 93.2% positive is consistent with this).
  These are different summary statistics of the same data, both verified; they
  are **not** in conflict (adjudicator checked: 74 × ~+0.31 for the ~93%
  positive events ≈ +21 nats, cancelled to +0.80 by ~5 sharply-peaked events).
  Both components push up; C5 has two contributors, not one.
- **Crossing-of-runaways framing: sustained and sharpened.** Dark-only argmax
  0.640, in-cat-only 0.860 in 10/10 runs, combined 0.700–0.742 (idealized:
  0.600 / 0.730 = truth / 0.730). Class slopes at the MAP are 5× smaller and
  total curvature ~1000× smaller than idealized; dh*/dε leverage 1500–2400×
  idealized; a ±1/√N_class Poisson reweight moves the combined MAP by up to
  0.025 (0.12–0.51 σ_h) vs 0.0000 idealized; λ-scan: λ=0 → 0.635–0.644, λ=2 →
  +0.011..+0.049. The run-to-run MAP sd 0.006–0.008 is **not** evidence of
  robustness against class composition (the CRB file is byte-identical across
  realizations of a seed, so class membership never varied). Independent
  corroboration: the C9 counterfactual moves 1D 0.732 → 0.643, i.e. 1D
  centredness is contingent on the same mis-calibration.
- Attacker's hole, noted: the leg-split (82%/18%) is r1-only, and their own
  leverage finding (near-flat combined profile) is precisely why *any* single
  counterfactual MAP displacement — including ones they and others report —
  must not be read alone as ownership. Applied consistently below.

### C6 — attribution confounded, control never run → **FINDING (confirmed by Gate A1); resolution in flight**
sig0_control ran `generator_marginal` + point ⇒ no estimator control exists.
The 2×2 cell B is running (jobs 6101146/6101147, pre-registered). The five
reports sharpen what cell B will mean — see §3. Nothing in the five reports
resolves the confound locally; every mechanism statement below is "as
delivered by the #53 configuration".

### C7 — host-z kernel omits selection → **promoted to FINDING (measured), with corrected law; scope narrowed** [LOCAL, VERIFIED]
- **Confirmed as the mechanism for C5's catalogue-leg per-event rail**, by
  measurement of the code's own numerator (driver validated against
  `fixed_quad` at 0.0e0; kernel h-invariance 9.1e-16; no quadrature aliasing).
- **The claim's formula is wrong for this code**: correct law
  `h_eff/h_true = [1 + √(1 + 12(σ_z/z)²)]/2 → 3(σ_z/z)²` (the extra +1/z from
  the numerator window's z-proportional width; the claim's 8-in-the-sqrt /
  2(σ_z/z)² understates 1.35–1.5×). Corrected sentence: at σ_z/z = 0.25–0.49
  the inflation is **+16% to +49%, h_eff 0.85–1.11** (claim said +11–36%,
  0.81–0.99). **Rail threshold: σ_z/z > 0.256.**
- The σ_z→0 limit gate passes (shift ∝ (σ_z/z)², log-log slope 1.99,
  coefficient → 1.500·2 = 3), so the §7 fix cannot disturb #51.
- **Confronted with production, not just predicted**: observed in-cat
  ball-numerator tilt median +0.308 nats (93.2% positive) vs predicted
  +0.33..+0.39 at σ_z/z 0.35–0.65 — vs **−408 nats, 0% positive** for the point
  kernel. The production data *independently* implies σ_z/z ≈ 0.35–0.6, so the
  verdict does **not** rest on the stale local `z_error` column.
- **Scope narrowed, two ways:** (i) channel-common (`prior_num` multiplies both
  numerators identically) ⇒ **not a C3/C4 candidate**, consistent with the
  exonerated "z leg" entry — properly not re-opened. (ii) **Acts AGAINST the
  dark rail**: K > 1 always, and dark events sit at σ_z/z ≈ 0.10 (K = 1.03);
  the dark catalogue-leg preference for 0.60–0.64 requires bare impostor
  z_g/ẑ ≤ 0.83 — **foreground contamination, a separate, unexamined
  mechanism** (new open thread; the inversion is censored data, a hypothesis
  not a measurement).
- Holes, noted: the with-scatter leg used synthetic z_obs draws (the realized
  observed catalogue is cluster-only); single-host prediction vs ball-sum
  observation (medians/rail fractions are the valid comparison); the
  "interior at h≈1.12" statement is reconstruction, not a delivered posterior.
  Cell B, which uses the true cluster parent `z_error`, is the staleness-free
  magnitude check (§3).

### C8 — 2D posterior reparametrization-dependent → **promoted to FINDING (well-posedness defect); cause RELOCATED** [LOCAL, VERIFIED]
- Reproduced exactly: C-walk 0.81329 / 0.78107 / 0.74440 / 0.600 (claim:
  0.8133 / 0.7821 / 0.7438 / 0.600; ≤1e-3 = MAP-refinement convention);
  **1D bitwise invariant** across the whole sweep; s = −1 established in closed
  form in all normalization modes and both mass-kernel families; sensitivity
  d(MAP₂D)/d ln C = **+0.031 per e-fold**.
- **The claim's stated cause is refuted**: it is *not* "4D numerator vs 3D
  D(h)" — D, β_G, β_Ḡ and Σ_glob(with_bh) are all **mass-dimensionless**;
  mass-marginalising D alone cannot restore invariance. The mismatch is
  **between the two numerator legs**: the 2D catalogue leg carries exactly one
  mass density (`mz_integral`), the completion leg carries none.
- **The claim's open question ("arbitrary vs fixed physical scale") is
  answered**: the code silently hard-wires the measure to dM_z/M_z,det,i — the
  **event's own** measured detector-frame mass (span 1.33e5–1.63e6 M☉, factor
  12; swapping in a constant of the same geometric mean already moves the MAP
  by 0.0056). A *consistent* physical unit change M → kM of all inputs is
  **exactly invariant** — so the runbook §7 acceptance gate "2D MAP invariant
  under M → kM" is vacuous as written and **must be restated as
  measure-invariance: 2D MAP invariant under L_cat,2D → L_cat,2D/C for
  arbitrary C** (equivalently: both numerator legs carry the same mass-density
  dimension so C cancels event-wise).
- **Canonical fix identified and priced (indicative, NOT ratified):** give the
  completion leg its missing dark-host mass likelihood g_i(z) from the code's
  own population prior. g_frac median 0.135 ⇒ the completion leg is currently
  over-weighted ~7.4×. Decomposition: h-frozen g(0.73) (pure measure fix) moves
  2D 0.8133 → 0.7558 (−0.058, agrees with the constant-C sweep at C ≈ 0.135);
  the full g(h) adds a +19.0-nat population tilt and lands at **0.84917 —
  independently reproducing the exonerated HA endpoint 0.8492 to 3e-5**. HA's
  exoneration is thereby **upheld and decomposed**: wrong sign = (−0.058
  measure, right direction) + (+0.093 model-dependent mass-function/redshift
  term). The upward term is what deserves scrutiny before any /physics-change.
- Hole, noted: g_i evaluated at z_i(h), not through B_num's quadrature; the
  ±0.093 split is model-dependent (Babak M1). The −0.058 measure part is
  robust (two independent routes agree).

### New numbered claims to add (all [LOCAL, VERIFIED] unless noted)

**C9 — w_G is mis-calibrated 2.3–2.5× against the code's own generator; the
inference's largest measured lever.** Model w_G(0.73) = 0.1215037 vs realized
detected in-cat rate 164/3135 = 0.05231 (76/1590 + 88/1545), binomial
**z = −11.86** pooled. Localized: the whole discrepancy is the catalogue's
relative detection efficiency — β_G weights f(z) by the **pool-marginal**
(population-mass) p_det, but Malmquist-selected catalogue hosts carry heavier
M–σ BH masses (rate-weighted median log₁₀M 6.9; ≥88% of rate weight above the
1e7 M☉ population cap by z ≈ 0.3). Two independent suppression measures agree
to 0.2σ: r(0.73) = Σw_Dg(with_bh)/Σw_Dg(no_bh) = 0.39248 (run's own logs) vs
realized 0.3991 ± 0.0312. Mass-aware w_G = 0.05149 → z = +0.21. Pointwise
p_det-free check: P(in-cat|det,z) matches f̄(z) below z = 0.15, then collapses
(1/62 in [0.15,0.20], 0/2882 above 0.20 where f̄ is still 0.11–0.29).
Counterfactual β_G → r(h)β_G (diagnostic, NOT a ratified fix): 2D mean 0.8123 →
0.7433 (bias +0.077 → +0.013), channel difference +18.80 → +11.36 (−7.44 nats,
40%), N·Δln(1−w_G) +31.55 → +12.10 — and 1D 0.7321 → 0.6430. **Scatter- and
realization-independent** (the host list is fixed by the CRB; the tension exists
identically in the idealized venue) — but absent from #51's *delivered* numbers
because `generator_marginal`'s w_G slot is a different, mass-aware estimand
(0.0555 at truth). This also **resolves the claim file's "loose thread"**: the
ghost values 0.0697/0.0686 lie on the generator_marginal curve
(0.0774/0.0692/0.0555/0.0427 at h = 0.60/0.64/0.73/0.86); the "45% discrepancy"
compared incompatible estimands. The #51→#53 switch changed the delivered
mixture weight by ×2.19 in the direction that inflates the prefactor tilt.
*Adjudicator's discounts:* (i) the "removes 84% of the bias" counterfactual
must be read against C5's leverage finding — in a near-flat profile many
±10-nat interventions move the MAP a lot; what is solid independent of leverage
is the **z = −11.86 generator-vs-inference inconsistency** and the two-way
0.392/0.399 agreement. (ii) The dark-side self-consistency (ε_dark = ε̂_dark)
is argued from construction, not measured (cheap follow-up listed). (iii) The
attacker's own cell-B sentence ("if the normalization mode owns it, B's 2D MAP
should come back near 0.73–0.75") is **inverted relative to their own
mechanism** — cell B runs `absolute_marginal`, so the mass-blind w_G = 0.1215
is present in B and the mechanism predicts B biased HIGH; see §3.

**C10 — the completion-channel up-pull is prefactor-carried.** Over 0.73→0.81:
N·Δln(1−w_G) = +31.55 (dark share +30.04, in-cat +1.51) while
ΣΔln L_comp = −3.11 (**dark −22.72, in-cat +19.61**); only 39.1% of dark
events have a positive completion tilt. Any sentence of the form "the
completion term pulls up" must name the (1−w_G) prefactor, not L_comp.

**C11 — completion-leg deep-venue calibration is an order of magnitude too
small to own the 2D bias.** pp_coverage extension to comp_frac 0.008–0.234
(landing #53's w_G ≈ 0.12 venue): bias +0.0008..+0.0097 at comp_frac 0.06–0.09,
+0.0034..+0.0181 at 0.13–0.24; monotone in comp_frac across the full 0.008–0.85
range, no sign flip, control-consistent at zero. 6–16× below +0.077. Live as a
modest contributor to the 1D +0.017 Option-A residual (same order); harness is
1D-only/single-channel by construction.

---

## 2. Cross-report reconciliations (adjudicator's synthesis)

1. **"Rail in the catalogue leg" (C7 report) vs "rail imported from the
   completion admixture" (C5 report):** both correct, different statistics.
   Per-event argmax (C5's literal statement) is catalogue-leg-driven and
   C7-explained; class-summed displacement is majority completion-admixture
   because a few golden events nearly flatten the catalogue class sum. Verified
   mutually consistent. The claim-file rewrite must carry **both**.
2. **The two "mass-aware corrections" are different objects and both lower the
   2D MAP** — a warning, not a contradiction. C9's r(h) *shrinks the catalogue
   prefactor* (reducing the (1−w_G) tilt, 0.81→0.745); C8's measure part
   *boosts the catalogue leg 7.4× against completion* (restoring dark
   down-pull, 0.81→0.756). They act on different terms of the same mixture,
   are **not additive**, and the full C8 fix's population tilt pushes the other
   way (+0.093). ⇒ **No piecewise patch is interpretable; the only coherent
   fix is a single jointly-derived mass-consistent mixture** (§4, gated).
3. **The exoneration list survives intact.** Nothing was re-opened. Two entries
   are strengthened: HB (its hard-zeros are worth 1.5% of the target —
   corroborates its self-refutation) and HA (endpoint independently reproduced
   to 3e-5, now decomposed). One Gate C formulation is retired: "D(h) never
   mass-marginalised" is refuted *as formulated* (D is mass-dimensionless);
   the live object is the completion **numerator's** missing mass density.
4. **A method bug to propagate:** the h grid is non-uniform (0.01 on
   [0.60,0.65] and [0.80,0.86]; 0.005 on [0.65,0.80]). Any second-difference
   across the seams is invalid. (The claim file's Gate-A3 check sat on the
   uniform part and is fine.)
5. **Where the weight of evidence now points.** Three *measured internal
   inconsistencies* sit on the completion/prefactor/kernel side — C8's missing
   mass measure, C9's z = −11.86 calibration, C7's selection-free kernel —
   while the mass de-weighting of impostors (the C4 carrier) is the 2D
   channel's *intended function*. The open question "which leg is wrong" is
   thereby materially sharpened: nothing yet convicts the catalogue leg, and
   three convictions stand against the completion-side stack. This is a
   direction, not a verdict; it is what cell B plus the joint fix derivation
   must settle.

---

## 3. The 2×2 cell B — mechanical readout (jobs 6101146/6101147)

Everything relevant in B is now understood to be **scatter-independent or
scatter-inert by prior measurement**: the volume_deconv width is the catalogue
`z_error` *column* (present in the unscattered parent — and B uses the true
cluster parent, making it the staleness-free C7 test); w_G is pure quadrature
(pre-registered bit-identical); the mass de-weighting compares near-exact GW
masses to catalogue masses whose errors are catalogue-intrinsic (the ln-M draw
moved masses ≤0.0009 dex); candidate-window membership is exonerated (0.81→0.82,
wrong sign). **The five reports therefore jointly predict the "estimator owns
it" outcome: B ≈ C in both channels** (2D ≈ 0.78–0.82, in-cat class argmax ≈
0.86, 1D ≈ 0.70–0.74 as a crossing). Register that as the expectation *now*, so
a contrary result is unmistakably a surprise.

- **Outcome 1 — estimator owns it (B ≈ C; the joint prediction).**
  The realistic host-observation model (the *scatter*) is largely exonerated
  for the headline numbers; the #53 **estimator configuration**
  (volume_deconv's selection-free population weight + absolute_marginal's
  mass-blind w_G + the un-measured completion mass density) made them. A→B
  isolates the estimator cleanly, including the destruction of the golden
  events' curvature (kernel-width, not scatter). Consequences: C7's fix and
  the joint C9+C8 mixture fix become the whole fix surface; #53's realism
  layer needs no revisiting for the bias; C5/C7 magnitudes get their
  staleness-free confirmation from B's in-cat L_cat argmax distribution
  (C7 Leg-A predicts ~99% above 0.86 at the parent's widths).
- **Outcome 2 — scatter owns it (B ≈ A: 2D ≈ 0.730, in-cat argmax ≈ 0.730).**
  A deep surprise that **falsifies the transfer of two confirmed mechanisms to
  the delivered posterior**: C7's kernel shift (scatter-independent by
  construction — its failure would indict the production confrontation, e.g.
  the ball-sum aggregation) and C9's lever (w_G = 0.1215 present in B yet no
  bias — implying strong unscattered catalogue legs mask the prefactor tilt,
  which would in turn mean the dark L_cat support structure differs between B
  and C far more than the ln-M-draw and membership exonerations allow).
  Mandatory follow-up in this branch: re-examine those two exonerations for a
  support-structure loophole before anything else.
- **Outcome 3 — mixed.** Read B's per-class summed profiles (secondary
  pre-registered reads): B−A in nats per class = estimator share; C−B =
  scatter share. Apportion C7/C9 (both in B−A) vs residual-scatter effects
  (C−B). If B's w_G ≠ 0.1215 bit-exactly, that is its own finding and voids
  the C9 transfer arithmetic for B.
- In all outcomes: B's in-cat **L_cat-leg** argmax distribution adjudicates C7
  independently of the MAP, and B's diagnostics (if enabled) would give the
  first non-r1 C3/C4 partition for free. If the job was submitted without the
  diagnostics CSV, pull the four 2D h-point JSONs (A2 recipe) instead.

---

## 4. Gate C sweep verdict — ranked bias-owner candidates

**Channel-common vs channel-specific (the precise form, replacing the loose
heuristic):** per event, p_ch = C(1+R_ch) with C = (1−w_G)L_comp common. A
perturbation cancels *exactly* from the channel difference only if it
multiplies C at fixed R. Perturbing **w_G** or **L_comp's calibration**
changes R_1D and R_2D differently wherever L_cat differs between channels —
so "channel-common" factors CAN own part of a channel difference **through the
channel-specific L_cat support** (31.0% of events are completion-only in 1D vs
61.8% in 2D; for the dark class R is small, so the cancellation is a good
approximation there and C11-type calibration still cannot own the difference).

### (a) Owners of the 2D +0.077

| rank | candidate | status | channel arithmetic |
|---|---|---|---|
| 1 | **C9 w_G mass-blind mis-calibration** | measured defect (z = −11.86), largest lever: owns −7.44/+18.80 (40%) of the channel difference and counterfactually 84% of the bias | common in origin; differential via L_cat support asymmetry |
| 2 | **C4-amended mass de-weighting of dark survivors** | the confirmed *carrier* (+15.60 of +15.83); whether it is defect or intended physics is THE open question | fully channel-specific |
| 3 | **C8 measure defect / completion mass density** | confirmed well-posedness defect; not a signed owner (−0.058 measure + ~+0.093 population = +0.036 net, wrong sign; upholds HA exoneration) | channel-specific (2D leg only) |
| 4 | **completion-leg deep-venue calibration (C11)** | REFUTED as 2D owner: ≤ +0.018, 6–16× short; and ~cancels from the dark channel difference | channel-common, no differential lever of its own |
| 5 | **HB residual** | already refuted; its h-flat suppression *is* rank 2's de-weighting scale, subsumed | channel-specific |
| — | **C7 host-z kernel** | not a 2D-difference candidate (channel-common through both numerators) — exoneration respected | channel-common |

Ranks 1–3 are **coupled through the same mixture** and their counterfactuals
are non-additive in a near-flat likelihood (§2.2). No single owner should be
declared; the decidable object is the jointly-derived mass-consistent mixture.

### (b) Owners of the 1D railed-crossing structure

1. **C7 kernel shift** — confirmed mechanism for the in-cat catalogue-leg
   per-event rail (the dominant per-event effect).
2. **Completion-admixture rise for nearby in-cat events** — carries ~82% of the
   class-summed in-cat mixture rise; plausibly correct completeness physics
   (1−f small nearby), **UNDETERMINED**; not explained by C11's small
   calibration bias.
3. **C9 w_G level** — shapes the crossing point (counterfactual 1D 0.732→0.643;
   1D centredness is contingent, treat as upper-bound sensitivity).
4. **Dark foreground-impostor down-pull** (z_eff/ẑ ≤ 0.83, 80.8% censored at
   the 0.60 edge) — the *other* runaway; mechanism unexamined. **New open
   thread**, sonnet-grade: Malmquist + volume weighting (correct physics) vs
   defect.

---

## 5. Next steps, cost order (runbook §7 routing respected)

1. **Free / this session:** write the claim-file edits (§6). Commit the
   `realistic_20260729/` analysis artifacts (scripts, JSONs, READMEs,
   adjudication; the 60 MB posterior pulls per author's call).
2. **Instrumentation, plain GSD — justified NOW, regardless of cell B:**
   per-class per-h Σln p split by `host_galaxy_index >= 0` (both channels);
   diagnostics CSV on **every** run (the r1-only caveats on C3/C4/C8 all trace
   to its absence); w_G at 7 s.f.; P6 counter with index-space translation.
3. **Cell B readout when it lands — mechanical per §3.** Cheapest decisive
   information in flight; nothing below should pre-empt it.
4. **Cheap local follow-ups (haiku/sonnet):** (i) dark-draw self-consistency:
   detected dark z-distribution vs β_Ḡ's integrand (closes C9's caveat 5);
   (ii) the dark foreground-impostor thread (§4b.4); (iii) if cell B has
   diagnostics, re-run the C4 partition off-r1.
5. **C7 fix (`/physics-change`): derivation can start now; ratify after
   cell B.** Justification is already sufficient on mechanism grounds (measured
   law, σ_z→0 gate passes with coefficient 3, production confrontation) — but
   the delivered-impact attribution and the staleness-free magnitude check
   arrive with B for free. Acceptance gates per runbook §7 plus: reproduce the
   measured [1+√(1+12e²)]/2 law before the fix, its collapse after.
6. **Joint C9+C8 mass-consistent mixture (`/physics-change`) — GATED on cell B
   and on the author's leg-adjudication.** Do **not** patch piecewise (§2.2).
   Acceptance gates, all now measurable: (i) **measure-invariance** — 2D MAP
   invariant under L_cat,2D → L_cat,2D/C (restated from the vacuous "M → kM";
   1D bitwise anchor already holds); (ii) **generator-calibration** — predicted
   detected-in-cat share consistent with the realized 164/3135 (Poisson);
   (iii) idealized #51 (`generator_marginal`) numerically undisturbed;
   (iv) an explicit derivation-time decision on the 1D numerator's mass
   treatment (1D **will** move; its current centredness is contingent — C5/C9).
   Note C9's z = −11.86 is defensible as a defect *today* independent of the
   bias question; only the **form** of the fix waits.
7. **Deprioritized:** HB `MASS_WINDOW_MODE` A/B (HB refuted; the de-weighting
   question belongs to gate 6's derivation); any GPU re-sim (nothing here
   needs one).

---

## 6. Edits to write back into `CLAIM_2D_BIAS_20260730.md`

Do not edit history silently; add a dated "Gate B/C adjudication" block and
strike-through, per the file's own convention.

1. **Header weakness paragraph (⚠):** strike "every 2D per-event number is
   [AGENT] and currently unreproducible" — the diagnostics CSV is re-staged at
   `seed61000/real_r1/diagnostics/event_likelihoods.csv` and all [AGENT]
   numbers reproduced exactly (`attack_c3_c4.py`; adjudicator re-ran). Replace
   with: "r1 is still the only realization with per-event 2D data."
2. **C3:** retag [AGENT, NOT REPRODUCED] → **[LOCAL, VERIFIED]**; status →
   FINDING; numbers unchanged; add r1-only-partition caveat.
3. **C4:** split. C4-obs (the measurement list) → **[LOCAL, VERIFIED]**,
   FINDING. C4-mechanism → **REFUTED AS STATED** (ln C cancels exactly, 6e-13;
   zeroed events carry 1.5%, survivors 98.5%) and replaced by the amended
   de-weighting mechanism with the exact budget (+15.83 = 0 + 19.10 − 3.27;
   f̄_dark 0.0354→0.0061; dark opposition −24.46→−0.63; dark argmax 0.785 vs
   completion-leg 0.810). Strike "fall back on the completion term, which pulls
   up" (also in the one-paragraph summary) and point to new C10.
4. **C5:** status → FINDING (replicated 10/10; edge-artifact refuted:
   validated-extrapolator peaks 0.93–1.05, C7 extended-grid interior ≈1.12 —
   clipped real runaway). Rewrite interpretation with the two-component
   attribution (per-event rail = catalogue leg, C7; class-summed rise ≈82%
   completion admixture) and the fair framing (median per-event displacement
   0.30–0.47σ but class-level +3.4–6.1σ; 54–67% edge vs 2.4% flat). Replace
   the corroborating "[AGENT] in-cat σ_h 3.2e-4 → 2.7e-2" with the measured,
   defined values (per-event σ_h 0.235–0.311 vs 0.043–0.053 idealized;
   σ_class 0.043–0.170). Keep "not a centred measurement"; add the leverage
   numbers (λ-scan, Poisson reweight, dh*/dε ×1500–2400).
5. **C6:** append Gate A1 result (sig0_control = generator_marginal + point ⇒
   CONFIRMED); cell B running (jobs 6101146/6101147), pre-registration file
   linked; add §3's joint prediction ("estimator owns it") as a dated
   pre-readout statement.
6. **C7:** retag → **[LOCAL, VERIFIED — MEASURED]**; status → FINDING
   (mechanism for C5's catalogue-leg rail). Replace the law
   (12-in-sqrt / 3e²; +16%..+49%, h_eff 0.85–1.11; rail threshold e > 0.256);
   add the σ_z→0 gate pass, the production confrontation (+0.308 vs
   +0.33..+0.39, point-kernel −408), the channel-common scope note, and the
   dark-direction result (K > 1 always; dark rail = foreground impostors —
   new open thread).
7. **C8:** retag → **[LOCAL, VERIFIED]**; status → FINDING (well-posedness).
   Replace the cause ("4D numerator vs 3D D(h)" → numerator-leg mismatch;
   D/β_G/β_Ḡ are mass-dimensionless). Answer the "arbitrary vs physical
   scale" question (implicit per-event dM_z/M_z,det measure; consistent unit
   changes exactly invariant; constant-scale swap moves 0.0056). Add
   dMAP/dlnC = 0.031/e-fold, the canonical g_i fix sketch (g_frac 0.135,
   completion over-weighted ~7.4×), and the HA decomposition (0.84917 = HA to
   3e-5 = −0.058 measure + +0.093 population). **Also fix runbook §7's HA
   acceptance gate**: "M → kM" → measure-invariance (L_cat,2D → L_cat,2D/C).
8. **Add C9, C10, C11** as numbered claims with the values in §1 above
   (C9/C10 [LOCAL, VERIFIED], r-independent where stated; C11 [LOCAL,
   harness]).
9. **Exonerated list:** append "HA — upheld and decomposed (−0.058 measure +
   +0.093 population tilt; endpoint reproduced to 3e-5)"; append "'D(h) not
   mass-marginalised' as a *formulation* — refuted (D is mass-dimensionless;
   live object = completion-numerator mass density, see C8)". Do **not** add
   w_G (C9 is live, gated on cell B).
10. **Errors made this session:** add **#6** — non-uniform h grid (0.01 on
    [0.60,0.65]∪[0.80,0.86], 0.005 on [0.65,0.80]); second differences across
    the seams are invalid. Add **#7** — the "loose thread" 0.0697 and
    ideal_61000.csv's 0.0686 are the mass-aware `generator_marginal` w_G
    estimand, not corrupt values; the "45% discrepancy" compared incompatible
    estimands (resolved into C9).
11. **"Loose thread, unexamined" section:** replace body with "RESOLVED →
    C9" and the one-line reconciliation.
12. **"What is explicitly NOT claimed" #1:** keep, append: "however, three
    measured internal inconsistencies (C7 kernel selection omission, C8
    missing mass measure, C9 w_G calibration z = −11.86) all sit on the
    completion/prefactor/kernel side; none convicts the catalogue leg. The
    leg-adjudication is now an evidence-weighted question, not an open coin."

---

*Artifact note: this file (`ADJUDICATION_20260730.md`) is the Gate B/C
adjudication of record for 2026-07-30; it is the only artifact produced by the
adjudicator. All attacker artifacts are described in `README_c5_c4_attack.md`,
`C7_README.md`, `README_C8.md`, `README_gateC_1_4_wG.md` (+ its `README.md`
duplicate), and `extract_and_compare.py`'s header in this directory.*
