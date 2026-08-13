# Pre-registration — MECHANISM ISOLATION: which term of the estimator produces the +1×σ_z displacement?

**REGISTERED 2026-08-13**, BEFORE any arm is run, on the author's verbatim approval
("all approved") of this file, of the V-M5 re-registration in §5, and of the arm code forms
pinned in the companion `ARMS.md`. Append-only from this commit: verdicts append below the final
rule, and no line above it may be edited.

Registered 2026-08-13, BEFORE any arm is run. Per RUNBOOK-9 successor; opens the `/physics-change`
intake authorised by the author 2026-08-13 ("I want to open the physics change"), ledger row #99.

## 0. Why this test is necessary

The venue-transfer campaign (prereg `e77eecad`, readout `d45fbf15`, ledger row #99) established
**that** the estimator is displaced and **how it scales**, at 1,400 seeds and ~160σ:

- decision cell T-c(0.730), N = 400, 1D: MAP bias **+0.037237 ± 0.000230**, HPD 50/68/90 coverage
  **0.000/0.000/0.000**, PIT–KS D = 1.000, rails 0.000, post_sd median 0.004376 (displaced by 8.5×
  its own claimed width), R_dose = 0.891;
- the ladder found **no killing axis** — real events, real multiplicity and real GLADE σ_z each
  left the collapse intact;
- the σ_z = 0 anchor T-0 put all 200 seeds exactly on truth.

It did **not** establish **which term** produces it. The `/physics-change` gate requires a *new
formula*; we can write the old one and cannot yet write the new one. This study exists to fill
exactly that slot, and nothing more: **it proposes no repair and adopts no candidate.**

### The three constraints every candidate must satisfy

Derived from the campaign, and used here as falsifiers rather than as support:

- **(a) vanish identically at σ_z = 0** — T-0 measured zero bias over 200 seeds, and at σ_k = 0
  the estimator takes a separate point-evaluation branch (`venue_transfer.py:1148-1165`).
  **Registered caveat (added 2026-08-13 from the M4 investigation): (a) is a weak discriminator for
  any mechanism entering as a smooth per-h function.** T-0's log-posterior peak curvature is
  1.73e8 against 7.35e4 at T-b, so the ~2,645 nats/h needed to cure the dosed bias would displace
  T-0 by only 1.5e-5 — invisible. T-0's perfection is evidence the *apparatus* is sound; it is not
  evidence that any particular per-h term is correct. Constraint (b) does the discriminating work;
- **(b) be linear in σ_z** — R_dose = bias/σ̄_z is 1.0688 (dose 0.011, v2 B1), 1.0075 (dose 0.035,
  v2 B2), 0.877–0.913 (GLADE mix). A σ_z² mechanism would differ by ~3× across those doses;
- **(c) not be misspecification** — generator and estimator share each candidate's σ_k by
  construction, so a systematic displacement is a handling defect.

### Candidate register at registration time

| id | term | status entering this study |
|---|---|---|
| **M4** | α(h) is σ_z-blind: numerator smears each candidate over a σ_z-wide kernel, the normalising denominator is built with no reference to σ_z (`closed_loop_gfrac.py:374-383`) | **CLOSED — REFUTED 2026-08-13**: the "missing" term is identically 1; see §7 |
| **M2′** | missing measure/Jacobian **inside** the z-integral, acting at the integrand peak (`venue_transfer.py:1138-1141`) | OPEN — raised by the M3 investigation 2026-08-13 |
| **M5** | flat 1/K candidate prior applied to candidates read at their *scattered* positions | **REFUTED AS STATED 2026-08-13** — causal attribution fails; see §7 |
| **M5′** | the estimator's own σ_z kernel builds an effective candidate measure **over-broad by 2σ²** (half from the data scatter, half from re-smearing it), times the marginalisation Jacobian τ = D/D′, against a **box** support — the ±4σ_d window, whose log has zero interior curvature and all curvature at its edges | **OPEN — the carrier.** Reproduces the defect in a validated toy; see §7 |
| **M1** | bare kernel 𝒩(z; z_obs, σ) used as p(z_true|z_obs) with no w_pop prior; w_pop appears in α(h) but not in the numerator | **REFUTED AS SOLE MECHANISM 2026-08-13 — WRONG SIGN**, retained as a compounding *negative* term; see §7 |
| **M3** | h-dependent truncation limits, kernel never renormalised over the retained domain | **CLOSED — REFUTED 2026-08-13**, see §7 |
| **M2** | missing Jacobian in the *point* distance term | **CLOSED — REFUTED**: would bias T-0, which is clean |

## 1. The run

- **RUN_DIR:** `results/mechanism_study_20260813/`
- **Arm code forms:** `results/mechanism_study_20260813/ARMS.md`, registered with this file.
- **Instrument:** `darksiren_emri/validation/venue_transfer.py` at the registration commit, plus
  one new term-ablation switch per arm (**estimator-side only**; the generator is untouched, so
  every arm consumes the identical seed realisation).
- **Inputs, carried verbatim from the venue-transfer campaign and re-verified before any arm:**
  CRB CSV md5 `9a1f2a14384a9281c97ca3be312ddaab`; frozeng emit md5
  `34c50e91028b6a6458a2b145db545705`; K census 1588/606/982/ΣK 1,193,703/max 245,364; pruned-frame
  σ_z stats n = 20,834,171, median 0.0393412950539589, min 0.0005317263419419, n<5e-3 231,098.
- **Base configuration for every instrument arm:** the campaign's decision cell — pinned 982
  events, real K_i, GLADE-empirical σ_z sampler, h_true = 0.730, canonical 41-point grid.
- **Code change note:** this study **does** change code — that is its purpose. Every change is an
  additive, default-off switch in the validation module. **No production module
  (`bayesian_statistics.py`, `physical_relations.py`, `constants.py`,
  `simulation_detection_probability.py`) is modified by this study.** Any production change is a
  separate `/physics-change` package downstream of this study's verdict.

### Seed plan (VT-D7 discipline carried)

New disjoint decade, base 20260808: **+50000…+50999**. Per arm at layer L1: 25 seeds; the L2
confirmation arm: 200 seeds. No seed here appears in v1 (+0…9049), v2 (+20000…29049), v3
(+40000…45199), or the reserved-and-unconsumed W1 (+46000…46399) / O2 (+47000…47399) blocks.
Unit-tested before any arm runs.

### Layered design — the cost argument

Per-seed bias is +0.037 against a per-seed MAP spread of ~0.005, i.e. **≈7σ in a single seed**.
Discriminating "term removed" from "term intact" therefore does not need N = 400.

- **L0 — analytic + toy isolation (CPU-minutes).** For each candidate, an A/B toy that differs
  *only* in the term under test, so everything else cancels analytically. Reports the implied MAP
  displacement and its σ_z-scaling. A candidate whose L0 implied displacement is **below 1e-3 in h**
  (a factor 37 under the observed effect, and 5× under the per-seed noise floor) is **CLOSED at L0
  without an instrument run** — this is what retired M3 for a few CPU-minutes.
- **L1 — instrument confirmation, N = 25, one arm per surviving candidate.** ~95 CPU-h/arm at the
  campaign's measured 3.79 CPU-h/seed.
- **L2 — single confirmation arm, N = 200**, run *only* for a candidate that L1 classifies
  TERM-OWNS, and only on author order.

Registered budget ceiling: **L0 unlimited (toys), L1 ≤ 5 arms, L2 ≤ 1 arm.** Exceeding it is a
STOP-and-consult, not a silent extension.

## 2. The design matrix

Every arm is the base configuration with **exactly one** estimator term altered. Arm N-0 alters
nothing and must reproduce the campaign.

The matrix below was **restructured 2026-08-13** after the L0 layer closed three of the five
original candidates and identified M5′ as the carrier (§7). The study is now primarily a
*confirmation* of M5′ on the instrument, plus one open alternative.

| arm | what varies | prediction |
|---|---|---|
| **N-0** (null) | nothing — the campaign path | bias **+0.037 ± 0.002**; anything else voids the study |
| **E1-host** | σ_z applied to the **host only**; every impostor keeps its exact redshift | bias ≈ **+0.006** if M5′ carries |
| **E1-imp** | σ_z applied to the **impostors only**; the host keeps its exact redshift | bias ≥ **+0.030** if M5′ carries |
| **A-M2′** | the measure/Jacobian restored inside the z-integral | bias → in-band if M2′ carries instead |
| **E3** | extended dose ladder at the crossover (σ_z vs the window half-width in z) | R_dose trajectory matching the toy's 1.16/0.85/0.75/0.72/0.76/0.88 |
| ~~A-M4~~ | *withdrawn* — the term is identically 1 (§7) | — |
| ~~A-M5b (W1)~~ | *withdrawn* — closed as a null **and** shown to double-count (§7) | — |
| ~~A-M1~~ | *withdrawn as an arm* — wrong sign; retained only as the fitted quadratic term (§7) | — |

**E1 is the decisive arm and it requires ZERO estimator change.** The per-candidate σ vector
already exists (`venue_transfer.py:1139`); only the *generator-side* assignment at
`venue_transfer.py:1393, 1396` varies between the two cells. This keeps V-M2 (generator
invariance) trivially auditable — the estimator is byte-identical across N-0, E1-host and E1-imp.
Registered cost: **2 cells × 15 seeds** (the split is predicted at ~5σ separation per the toy's
+0.0247 vs +0.0062 at K = 50).

**DS-M5 — the split-dose read (primary).** M5′-CONFIRMED requires **both**: E1-imp bias ≥ 0.030
**and** E1-host bias ≤ 0.012, at N = 15 (SE ≈ 0.0013). A split in the opposite direction, or both
cells large, refutes M5′ and returns the study to the M2′ arm.

The exact code form of each altered term is fixed in this file's companion `ARMS.md` **before
registration** and may not be adjusted afterwards.

## 3. Decision statistics — bands locked at this commit

Provenance of every number below: the venue-transfer prereg §7 rows (`e77eecad`), the campaign's
committed decision-cell values (`d45fbf15`), and binomial/Jeffreys arithmetic at N = 25 and
N = 200. **Nothing below uses any number produced by this study.**

**DS-M1 — MAP bias, per arm, 1D channel (the registered headline; 2D reported alongside).**
Edges carried verbatim from the campaign: in-band |b| ≤ 0.010; DEFECT |b| ≥ 0.030.
At N = 25 the SE on the mean bias is ≈ 0.0010 (from the campaign's per-seed sd 0.005), so all three
classifications below are separated by ≫ 10σ.

Per-arm mechanical classification:
- **TERM-OWNS** = |b| ≤ 0.010 **and** HPD90 coverage ≥ 0.60 at N = 25.
- **TERM-PARTIAL** = 0.010 < |b| < 0.030.
- **TERM-INNOCENT** = |b| ≥ 0.030 **and** |b − b_N0| ≤ 0.004 (within 2σ of the null arm at N = 25).
- **OTHER** = anything else — reported raw, direction stated, no branch forced.

**DS-M2 — HPD coverage at N = 25** (binomial, same formula as the campaign's DS-VT1):
2σ bands 0.500 ± 0.200 / 0.680 ± 0.187 / 0.900 ± 0.120. Reported for every arm; carries branch
weight only through the TERM-OWNS conjunction above.

**DS-M3 — dose-scaling of the residual.** Each surviving arm re-run at the two flat doses 0.011
and 0.035 (5 seeds each, L0 toy where the toy is faithful). Registered read: a term that OWNS the
defect must remove the *linearity*, not merely shrink the amplitude — residual R_dose must fall
below **0.25** at both doses.

**DS-M4 — W1's question, answered on the record.** Arm A-M5b is classified WEIGHTS-MATTER
(|b| changes by > 0.004) or WEIGHTS-INERT (|b| changes by ≤ 0.004). Either result closes the
dropped W1 arm's question; neither carries branch weight for the mechanism verdict.

**Anti-tuning.** Every threshold in this section (0.010 / 0.030 / 0.004 / 0.60 / 0.25 / 1e-3 at L0;
the N = 25 and N = 200 rows; the +50000 seed decade) is fixed at this commit and derived from
committed campaign artifacts or standard binomial arithmetic. None may be adjusted after any arm
is read.

## 4. Branches (presented to the author, never self-adjudicated)

Checked in order:

1. **STUDY-CONFOUNDED** — arm N-0 fails to reproduce the campaign bias within ±0.002, or any
   validity check in §5 fails. Every measurement below is void; author call on repair-and-rerun.
2. **SINGLE-OWNER** — exactly one arm is TERM-OWNS. That term is the identified mechanism; the
   `/physics-change` package is written against it, with this study's arm as its regression test.
3. **MULTI-TERM** — two or more arms are TERM-OWNS, or A-ALL owns while no single arm does. The
   defect is a conjunction; the gate package must address all contributing terms together, and the
   ladder of partial fixes is reported rather than a single culprit.
4. **NO-OWNER (first-class, non-forcing)** — every arm is TERM-INNOCENT or TERM-PARTIAL. The
   mechanism is not in the register. Handling, pre-stated: the register is exhausted, not the
   question; the next step is a fresh stage-0 intake with a mandatory Stage-L literature sweep
   (R0 ring at minimum) before any further arm is built. **No repair may be proposed from a
   NO-OWNER read.**

## 5. Validity checks and STOP criteria

- **V-M1 — null-arm reproduction.** Arm N-0 must reproduce the campaign's decision-cell bias
  within ±0.002. This is the study's own anchor; failure ⇒ STUDY-CONFOUNDED.
- **V-M2 — generator invariance.** For a fixed seed, `K_sum`, `event_idx`, the **pre-dose** `z_obs`,
  the σ_z vector and the standard-normal scatter vector must be **bit-identical across all arms**;
  only the dose mask differs. The **post-dose** `z_obs` necessarily differs between arms — that
  difference *is* the experiment. RNG order, draw count and stream are identical in every arm.
  Formalised as AR-1/AR-2/AR-3 in `ARMS.md` and unit-tested. Any arm that perturbs a *draw* rather
  than a *mask* is invalid by construction — the whole design rests on this.
- **V-M3 — pin integrity.** The §1 md5s, K census and σ_z statistics re-verified before any arm.
  Any mismatch ⇒ STOP.
- **V-M4 — clean rule.** Carried verbatim from the campaign (V-T4), with the import path read as
  `darksiren_emri/` + `darksiren_emri_test/` (the package was renamed in `227e7a32`, after the
  campaign; this is a wording update to the same rule, not a relaxation of it).
- **V-M5 — no-drift anchor, RE-REGISTERED AS A VALUES GOLDEN.** The campaign's V-T5 demanded
  *bit-identical* reproduction of the committed v2 `B2_h0p730` per-seed records. **That test now
  fails on current `main`** — verified 2026-08-13: `ln_post_2d` differs by max 2.2e-16 relative
  (≈1 ULP), `M_source_median` by 1.0e-15, `pit_2d` by 1.6e-14; the **1D channel is bit-identical**
  and **both channels' MAPs are unchanged**. Cause: the author-ratified Route 1 adaptive
  Gauss–Hermite change to the g_i contraction (`bayesian_statistics.py`, ledger 2026-08-12,
  certified max rel err 1.3e-15), which reaches the validation stack because
  `closed_loop_gfrac.py`, `calibration_gate.py` and `venue_transfer.py` all import from that
  module. Bit-identity is therefore **unsatisfiable by construction** on current main and would
  STOP this study for a non-defect.
  **Registered replacement:** V-M5 passes when every shared field agrees with the committed record
  to **rtol ≤ 1e-12** *and* both channels' MAP values are exactly equal. Precedent: commit
  `650605ad`, where the R1 gate pin was converted from an md5 digest to a values golden at
  rtol 1e-12 after numpy SIMD dispatch made the digest host-dependent — the same disease, the same
  cure. **This is a registered loosening of a prior STOP criterion and requires explicit author
  ratification before this file is registered.**
- **Abort criteria:** (a) non-finite ln_post in > 1 % of any arm's seeds ⇒ STOP; (b) horizon-drop
  > 5 % ⇒ STOP; (c) any V-M failure ⇒ STOP; (d) an arm whose L0 toy and L1 instrument disagree in
  *sign* ⇒ STOP and report — the toy is then not faithful and every L0 closure must be revisited.

## 6. Expected NULL results (pre-registered)

Stating these in advance so that a null is a *reading*, not an absence:

- **Arm N-0 vs the campaign:** expected to reproduce the bias within ±0.002. If it reproduces it
  *bit-identically* that is a bonus, not a requirement — the Route 1 change (V-M5) means bit
  equality is no longer expected anywhere the 2D channel is involved.
- **A-M1:** expected to move the bias by a σ_z²-scale amount, i.e. **not** to reach in-band. If it
  lands TERM-OWNS, constraint (b) is wrong and the dose-scaling evidence must be re-examined before
  anything is concluded.
- **A-M5b (W1):** the adjudicator's common-mode reading (finding D11) predicts **WEIGHTS-INERT**.
  If weights turn out to matter, the dropped W1 arm was more informative than judged on 2026-08-13
  and that judgement is recorded here as having been wrong.
- **2D channel:** expected to track the 1D classification in every arm, as it did in every cell of
  the campaign. A 1D/2D split in any arm is itself a finding and forces the MULTI-TERM branch.

## 7. Register of closures made before registration

**M3 — REFUTED 2026-08-13**, before this file was drafted, at a cost of a few CPU-minutes. Full
note: `results/mechanism_study_20260813/M3_truncation_window.md`. Summary of the falsification, so
that the closure is auditable rather than asserted:

The truncation edge is pinned at ±4σ **in the GW-likelihood variable**, not in the kernel variable —
by construction d_L(z_lo/hi, h)/d_obs = 1 ∓ 4σ_d at every h. Everything the window discards
therefore lives under e⁻⁸ of the integrand peak, capping the fractional perturbation to any
candidate's c₁ at 2Φ̄(4) = 6.3e-5 **however much kernel mass is clipped**. An A/B toy differing only
in the window width (12σ_d vs 4σ_d) measured per-event |Δ ln L| = 3.8e-5 (max 6.3e-5, hitting the
analytic ceiling exactly), implying a MAP displacement of **+6.0e-7 in h — a factor 6.2e4 short** of
the observed +0.0372. It also fails constraint (b) independently: implied shifts *decrease* with
dose (2.3e-6 / 8.1e-7 / 6.5e-7 at σ_z = 0.011 / 0.035 / 0.042), the wrong trend by ~13× across a
factor 4 in dose.

**Corollary carried into this study:** because the window is pinned at 4σ in the GW variable, *any*
mechanism confined to the p_gw wings is capped at ~1e-4 per event, and widening `_SIGMA_WINDOW` is
an inert knob. **A surviving mechanism must act at the integrand peak.** M2′ was raised by this
corollary and enters the register above.

**M1 — REFUTED AS SOLE MECHANISM 2026-08-13, on sign rather than scaling.** Full note:
`results/mechanism_study_20260813/M1_missing_volume_prior.md`. Expanding the Bayes-correct
E[z_true|z_obs] against a locally log-linear population prior gives Δz = σ_z²·λ, λ = d ln p_pop/dz.
On the venue's actual 982-event population (median z = 0.494, IQR [0.35, 0.63], decoded from the
pinned CRB CSV) λ ≈ 2.3–2.9 and is **positive throughout**, so M1 predicts H₀ biased **LOW** by
~0.02–0.04 at σ_z = 0.035–0.042 — the same order as the observed defect and the **opposite sign**.
It is quadratic as expected, failing constraint (b) independently.

**Retained as a compounding negative term.** A least-squares fit of bias(σ) = aσ + bσ² to the three
committed dose anchors gives a ≈ +1.15, b ≈ −5.29, with |b| within a factor ~2 of λ — the observed
R_dose *drift* (1.069 → 1.008 → 0.877–0.913) is exactly the signature of a dominant positive linear
driver plus a negative quadratic one. This is order-of-magnitude consistency, not proof, and it
makes **a ≈ +1.15 the quantity every surviving candidate must supply.**

**Pre-registered directional read for arm A-M1** (replacing the generic expectation in §6): if M1 is
genuinely a *subtractive* compounding term, restoring the w_pop prior must make the residual bias
**larger and more positive**, not smaller. An arm A-M1 that reduces |b| falsifies the compounding
account and re-opens M1 as a sign-ambiguous term.

**M4 — REFUTED 2026-08-13, analytically and at zero simulation cost.** Full note:
`results/mechanism_study_20260813/M4_alpha_sigma_blindness.md`. The term M4 claims is missing is
**identically 1**. α(h) is the marginal probability of *detection*, and detection depends only on
the GW observables (`closed_loop_gfrac.py:446-452`); the only self-consistent σ_z-aware form is
α_σ(h) = ∫dz w_pop S̄_φ · ∫dζ 𝒩(ζ;z,σ), whose inner integral is 1 because the code's kernel
(`venue_transfer.py:1139`) is a properly normalised density. **α's σ_z-blindness is correct, not a
defect.** The one escape — selection keyed on the noisy ζ rather than true z — is closed by
construction: ball membership is decided on true z (`calibration_gate.py:677-702`,
`venue_transfer.py:845-868`) and σ_z is applied afterwards (`venue_transfer.py:1393, 1396`), with
K_i pinned.

**Direct experiment, run on stored data with no new simulation.** Because α enters as a single
additive per-h function, the entire α family is testable by post-processing the campaign's stored
`ln_post` vectors. Rebuilding `log_alpha` from the campaign config and re-argmaxing all 2,400
posteriors with α **deleted outright**: bias +0.0353 → **+0.0165** at σ_z = 0.035, and
+0.0107 → **+0.0056** at σ_z = 0.010, remaining ≈ linear in dose. **The σ_z keying survives total
deletion of α.** α does contribute a σ_z-blind high-h pull (−N ln α = +1.036 N ln h; measured
d ln α/d ln h = −1.0358, power-law to 0.2 %) — it roughly *doubles* the amplitude but does not
*key* it. **Registered consequence: any candidate that OWNS the defect must survive this test —
it must still produce ≈ +0.0165 at σ_z = 0.035 with α removed.**

**THE PARITY ARGUMENT — the strongest constraint now in hand.** Gaussian convolution is
exp(σ²∂²/2), an expansion in **even powers of σ only**. Therefore *every* "we convolved wrong"
story — the whole kernel-mismatch class — is O(σ_z²) at leading order and predicts R_dose ∝ σ_z,
i.e. a 3.5× change across the B1→B2 dose lever. Measured: R_dose 1.103 → 1.012, ratio **0.92**
against a predicted **3.5**. **A surviving mechanism therefore cannot be a symmetric smoothing of
any kind.** It requires genuine first-order structure at scale σ_z — a support edge, the argmax
operation itself (which is not a smooth functional of the posterior), or a host/impostor asymmetry
inside the finite ball window. Noted in the M4 study: the ball-window half-width in z and σ_z are
**comparable at these doses**, which makes the ball membership boundary — a hard edge defined on
*true* z, crossed by candidates scattered with σ_z — the natural next object. Note this is a
different edge from the one M3 refuted: M3 tested the GW-likelihood truncation in the p_gw wings;
this is the candidate-membership support.

**M5 — REFUTED AS STATED 2026-08-13; M5′ IDENTIFIED AS THE CARRIER.** Full note:
`results/mechanism_study_20260813/M5_smeared_candidate_prior.md`. A ~120-line standalone toy
mirroring the estimator exactly (real D(z), d_L = D(z)/h, w_pop population + horizon α(h), per-event
σ_d resampled from the pinned CRB CSV, ±4σ_d window, K−1 impostors from w_pop|W, GL-50, ±5σ clip,
1/K, −745) **validates against T-0** (bias +0.00093 ± 0.00042 at σ_z = 0) **and reproduces the
defect** (R_dose 0.72–0.95 against the instrument's 0.88–1.07). It is the first object in this
thread that does both, and it is what makes the ablations below causal rather than suggestive.

*Why M5-as-stated fails* — attribution, not constraints: with the population **not scattered at
all**, 76 % of the bias remains, so "n ⊛ 𝒩(0,σ)" is neither necessary nor dominant; deleting the
window truncation changes the result by −1 % (the window is h-*independent* in the u-substitution,
so the "scattered out of the window" sub-claim is dead on arrival); and no prior or weight repair
attenuates.

*What survives (M5′).* Under the substitution ζ(h) = D⁻¹(h·d_obs), each candidate term collapses to
ℓ_k(h) = τ(ζ)·𝒩(ζ(h); z_k, Σ_k) with Σ² = σ_z² + s² and τ = D/D′. The estimator's own kernel builds
an effective candidate measure **over-broad by 2σ²** — half the data scatter, half re-smearing it —
against a **box** support whose log has zero interior curvature and all curvature at the edges, so
the MAP sits on the smeared upper flank at 4s + κ√(2σ² + s²). **This is the linear, first-order,
support-edge structure the parity argument demanded**, and it is why R_dose sits near 1 and drifts:
σ_z and the window half-width in z are comparable at these doses, i.e. the campaign ran at the
crossover between the √(2σ²+s²) ≈ s + σ²/s and ≈ √2σ regimes. Sign is positive on two independent
counts (d ln τ/dz > 0 and d ln n/dz > 0, with ζ rising in h); removing τ removes 32 % of the effect.
The Gaussian/Eddington σ² term exists and is **14× too small**.

*Split-dose evidence (the basis for arm E1).* At K = 50, dosing **impostors only** produces
**+0.0247** of the **+0.0334** total, while dosing the **host only** produces **+0.0062**. The
host's exact redshift is the only thing pinning the estimate at σ_z = 0, and its pinning power is
just 1/K — which is why T-a → T-b (K ≈ 5 → 1,216) moved the bias by only 0.001 and why the effect
saturates in K (K = 2/5/20/100 → 0.0138/0.0252/0.0314/0.0341).

**W1 — CLOSED AS A NULL, and it would have been the wrong repair.** The adjudicator's common-mode
reading (finding D11) is confirmed *exactly*: a displacement shared by all candidates sits inside
every kernel, so it factors out of **any** h-independent convex combination — the stationary point
moves by exactly δ for any K and any weights, and 1/K being h-independent drops from the argmax.
Measured in the toy: rate-shaped weights **+2 % worse**, oracle weights at true z **+1 % worse**,
w_pop inside the integral **+28 % worse**, window renormalisation **+22 % worse**. Structurally
decisive on top of that: ball impostors are *sampled from* w_pop|W (`calibration_gate.py:662`), so
equal weights **already carry the rate measure** — W1 would have double-counted it. The
2026-08-13 decision to drop the W1 arm is therefore vindicated on stronger grounds than it was
made: not merely redundant, but a repair pointing the wrong way. Seeds +46000…+46399 stay reserved
and unconsumed. **The §6 pre-registered expectation that a WEIGHTS-MATTER result would have made
that decision wrong is resolved in favour of the decision.**

**DISCRIMINATOR RAISED BY THE M1 CLOSURE — the pp_coverage sign flip.** `validation/pp_coverage.py`
carries the same structural bare kernel (`pp_coverage.py:868` vs `venue_transfer.py:1136`) and its
committed comparisons (`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`,
`results/pp_coverage_sigmaz_scan_20260703/`) show `bare` biasing H₀ **LOW** by −0.02 to −0.046
across σ_z = 0.005–0.05, with coverage collapsing to 0–3 %. Our venue, with a structurally identical
kernel, biases **HIGH** by +0.037. Two estimators, one kernel, opposite signs. What differs between
them is (i) the multi-candidate ball with its flat 1/K prior and impostors, and (ii) the α(h)
selection normalisation — i.e. **exactly the M5 and M4 terms**. Registering this as a pre-stated
cross-check: any candidate that OWNS the defect should also account for the sign flip between these
two venues, and an arm that removes the venue's positive bias while leaving pp_coverage's negative
one unexplained is a partial account at best. This cross-check carries **no branch weight** — it is
a directional sub-prediction, reported alongside the verdict.

---

*Verdict to be appended below by the session that reads out this study — after this file is
committed, no edits above this line.*


---

## Operational completion record — 2026-08-13 (NOT a readout, NOT a verdict)

Array **6303086** (`cluster/mechanism_isolation.sbatch`), partition `cpu_il`, 15 cores/task,
12 h requested. **All 3 tasks COMPLETED**, zero FAILED/TIMEOUT:

| task | arm | elapsed | seeds |
|---|---|---|---|
| 0 | MN0 (dose=all) | 00:58:08 | 20310808–20310822 |
| 1 | MEH (dose=host) | 00:02:20 | 20310908–20310922 |
| 2 | MEI (dose=impostors) | 00:56:30 | 20311008–20311022 |

**Probe-vs-actual pair (EXP-61 discipline).** `sbatch --test-only` predicted a start of
**2026-08-17T08:26** — four days out. Actual start was **within ~30 minutes of submission**, and the
whole array completed **inside 1 h**. The probe ignores backfill and is again wrong by orders of
magnitude for short wide jobs; second recorded instance after the venue-transfer campaign. The 12 h
request (3.8 h uncontended × the 1.9× contention factor) was ~12× larger than needed — the arms are
15 seeds rather than the campaign's 25, and the two heavy tasks shared an otherwise-quiet 128-core node.

**Retrieval.** `rsync -az --exclude='*.md'` — the exclude is deliberate and stronger than
`--ignore-existing`: it makes it impossible for a cluster copy to overwrite this registered file or
`ARMS.md`. (The venue-transfer retrieval earlier on 2026-08-13 silently reverted its own prereg with
a plain `rsync`; that is the incident this guard exists for.) Verified post-transfer: both `.md`
files show no modification in `git status`.

**Registered checks passing.** `K_sum = 1,193,703` on all 45 seeds across all three arms;
`n_events = n_events_run = 982`; zero rails in any arm or channel; zero non-finite `ln_post`.
Dosing verified per arm via `sigma_z_mean_pairs`: MN0 0.041813, MEI 0.041786 (impostors dosed),
MEH 0.000035 (= 0.0418 × 982/1,193,703 — exactly and only the hosts).

**Raw extraction (mechanical; no band scoring, no classification, no branch).**

| arm | ch | N | bias | SE | post_sd median |
|---|---|---|---|---|---|
| MN0 | 1d | 15 | +0.034667 | 0.001579 | 0.004265 |
| MN0 | 2d | 15 | +0.037000 | 0.001604 | 0.004315 |
| MEH | 1d | 15 | +0.004000 | 0.000535 | 0.000187 |
| MEH | 2d | 15 | +0.004333 | 0.000454 | 0.000262 |
| MEI | 1d | 15 | +0.000000 | 0.000000 | 0.000000 |
| MEI | 2d | 15 | +0.000000 | 0.000000 | 0.000000 |

**Two facts the readout session must confront (stated here, adjudicated nowhere).**

1. **The split is strongly non-additive.** MEH + MEI = +0.004 against MN0's +0.0347. The registered
   DS-M5 prediction (MEI ≥ 0.030, MEH ≤ 0.012) is *inverted on its decisive half*: the impostor-only
   arm carries **nothing**. Its posterior is not merely unbiased — it collapses onto a **single grid
   point** holding all the mass (MN0 spreads over 10), i.e. one exact host redshift overwhelms
   ~1,216 smeared impostors outright. Verified not to be an un-dosed arm: `sigma_z_mean_pairs` is
   full and `K_sum` is pinned.
2. **V-M1 fails as written.** Campaign decision-cell 1D bias +0.037237; MN0 measured +0.034667;
   |Δ| = **0.002570 against a registered ±0.002 window**, so STUDY-CONFOUNDED fires mechanically.
   Offered as disclosure and NOT as grounds to move the band: the N = 15 SE is 0.001579, so this is
   a **1.63 σ** deviation, and the ±0.002 window was registered without accounting for the arm's own
   sampling error — it is tighter than the statistic it gates. The 2D channel lands at +0.037000, on
   the campaign value. **The band may not be adjusted after a readout (§3 anti-tuning); it is
   recorded here as a design fault of this pre-registration, for the author's ruling.**

The scoring readout and the branch call are pending and belong to a separate session; nothing above
constitutes either.
