# BOOK_PEDAGOGY.md — the reader journey for *A Dark Siren Discovery Book*

**Role of this document.** Pedagogy architecture for the interactive discovery book:
personas and prerequisite ladder, the discovery arc chapter by chapter, the recurring
narrative beats, the full self-check question set with hidden answers and misconception
traps, and the interactivity specifications. It is a *build spec* — a writer and an
implementer should be able to work from it without re-deriving the design.

**Non-negotiable inherited constraint.** The book **cites** the project's ratified
derivations; it never re-derives loosely, never rounds a claim up into a certainty, and
the final chapters present the **currently known-inconsistent state (C7 / C8 / C9)**
honestly. Every number quoted below was checked against a repo artifact while writing
this document; provenance is given inline so the writer can carry it into the prose.

Status vocabulary is inherited from the project and used verbatim throughout the book:
**RATIFIED · CANDIDATE · OPEN · REFUTED · EXONERATED · CONFOUNDED**.

---

## Part 0 — Source-of-truth map for the writer

The book's authority chain, in the order a chapter should reach for it:

| Layer | Artifact | Book uses it for |
|---|---|---|
| Master equation | `bayesian_statistics.py:3006-3009, 1042-1048` | the one equation the whole book builds |
| Sky marginal of the completion term | `docs/derivations/G2a_completion_sky_marginal_4pi.md` | Ch 5, Ch 6 |
| Host-z volume prior | `docs/derivations/G2b_host_z_volume_prior.md` | Ch 7 (the σ_z² law) |
| Gray symbol mapping | `docs/derivations/G2c_gray_a9_a10_mapping.md` | Ch 3–5 (external anchor) |
| Host-mass rate prior | `docs/derivations/G2d_host_mass_rate_prior.md` | Ch 8 |
| 2D mass kernel (RATIFY-M6, **CANDIDATE**) | `docs/derivations/mass_marginal_2d_kernel.md` | Ch 8, Ch 11 |
| Realistic host-observation model (R1–R9) | `docs/derivations/realistic_host_observation_model.md` | Ch 9, Ch 10 |
| PV / photo-z kernel | `docs/derivations/hostz_pv_photoz_kernel.md` | Ch 7 |
| Inner-product dt² | `docs/derivations/G8_dt2_inner_product_derivation.md` | Ch 6 |
| Plunge window | `docs/derivations/plunge_window_initial_conditions.md` | Ch 9 |
| Normalization modes | `results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md` | Ch 9 |
| Adjudicated present state | `results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md` (as amended) + `gate_b_20260730/ADJUDICATION_20260730.md` | Ch 11 |
| 98-hypothesis history | `gate_b_20260730/BIAS_HISTORY_LEDGER.md` | the Defect Museum |
| Investigation ledger | `docs/H0_BIAS_RESOLUTION.md` (2666 lines) | Museum, mined selectively |
| External anchor | Gray et al. 2020, arXiv:1908.06050 | Ch 3–5 |

**Figure data that already exists and should be reused, not regenerated:**

- `results/campaign51_20260728/realistic_20260729/seed{61000,62000}/real_r{1..5}/diagnostics/event_likelihoods.csv`
  — per-event × per-h columns `w_G, L_cat_no_bh, L_cat_with_bh, B_num, L_comp,
  combined_no_bh, combined_with_bh` on a 41-point h-grid. **This one file family powers
  the majority of the book's interactives.** 1588 events × 41 h ≈ 65k rows per run.
- `.../real_r1/prepared_cramer_rao_bounds.csv` — 128 columns: the 14 EMRI parameters, the
  full Fisher/CRB covariance, `SNR`, `host_galaxy_index`, `in_catalog`, frame stamps.
- `.../posteriors/h_0_*.json`, `.../posteriors_with_bh_mass/…` — per-event 1D and 2D posteriors.
- Idealized baselines (campaign #51 `root`), the production injection pool
  (`injection_pool_mix200k_20260728`, 200,100 rows).

**Caution the writer must respect** (from the claim file's *Errors made this session*):
the root/idealized diagnostics CSVs contain **two concatenated evaluate sweeps** — do not
compute on them without era disambiguation; and the h-grid is **non-uniform** (0.01 on
[0.60,0.65] ∪ [0.80,0.86], 0.005 on [0.65,0.80]), so second differences across the seams
are invalid. Any interactive that differentiates numerically must be told about the seams.

---

## Part 1 — Personas, and what each chapter may assume

### 1.1 The three readers

**P1 — PRIMARY: "Mara", physics MSc student.**
Has: multivariable calculus, probability densities, Gaussians, Bayes' rule, a semester of
statistical methods. Has written a Metropolis sampler once. Has **no** GW astronomy, **no**
selection-effect formalism, **no** catalogue experience, and has never seen a likelihood
whose *denominator* is the interesting part. Wants to understand *why* each piece exists
and to leave able to smell a broken estimator.
**Design consequence:** the main narrative column is written for Mara end-to-end. Every
new symbol is introduced with units and a physical picture before it appears in an
equation. No chapter may require reading a linked derivation to follow the argument —
links deepen, they do not complete.

**P2 — SECONDARY: "Tomas", GW researcher.**
Knows Schutz '86, Gray et al. 2020, the `gwcosmo` family, LISA sensitivity curves. Wants
this pipeline's *specifics*: which equations are implemented, where they depart from
Gray, what the measured failure modes are, and which claims survive attack.
**Design consequence:** every chapter carries a collapsible **`▸ For the GW reader`**
stratum containing (a) the exact code site, (b) the departure from / mapping onto the
literature, (c) the measured numbers with their evidence tag. Tomas can read the book by
opening only these plus Ch 6, 8, 11.

**P3 — TERTIARY: "the examiner / collaborator".**
Wants to know what is *established* versus *claimed*, who ratified what and when, and
whether the honest state is being represented honestly.
**Design consequence:** every equation and every quantitative claim carries a
**provenance stamp** — a small badge rendering `RATIFIED (date, gate) · CANDIDATE ·
OPEN · REFUTED` with a hover card giving artifact:line. A chapter-level **Provenance
Panel** at the foot of each chapter lists every stamp used, so P3 can audit a chapter in
one screen. Ch 11 is written *for* P3 and is the book's honesty contract.

### 1.2 The persona switch

A single global control, **`Reading as: [Mara] [Tomas] [Examiner]`**, sets which strata
are open by default. It never *hides* content — it only pre-expands. Mara can open
Tomas's stratum any time and is nudged to at three specific points (Ch 6 Fisher, Ch 8
mass measure, Ch 11 adjudication). This is progressive disclosure by *reader*, not by
difficulty, which avoids the usual failure of "advanced boxes" being read as optional
decoration.

### 1.3 Prerequisite ladder

Cumulative. "New tool" is what the chapter *teaches*; "may assume" is everything from
strictly lower rungs.

| Rung | Chapter | New conceptual tool | May assume |
|---|---|---|---|
| L0 | Ch 0 | — (motivation only) | calculus, pdf, Gaussian, Bayes' rule |
| L1 | Ch 1 | waveform → d_L; the *dark* siren problem | L0 |
| L2 | Ch 2 | posterior over one parameter from N events; log-additivity; bias vs scatter | L1 |
| L3 | Ch 3 | marginalizing a **discrete latent** (which host); mixture with rate weights | L2 |
| L4 | Ch 4 | conditioning on **detection**; the selection denominator; p_det | L3 |
| L5 | Ch 5 | law of total probability over a **binary latent** (in/out of catalogue); the mixture weight as a posterior probability | L4 |
| L6 | Ch 6 | Fisher information, Cramér–Rao, multivariate Gaussian, change of variables, Parseval | L5 |
| L7 | Ch 7 | deconvolution; likelihood ≠ posterior; Eddington/Malmquist bias | L6 |
| L8 | Ch 8 | joint 2-parameter kernels; lognormal scatter; **dimensional consistency of a density (measure)** | L7 |
| L9 | Ch 9 | forward simulation; importance sampling; **generator–estimator consistency** | L8 |
| L10 | Ch 10 | frequentist coverage of Bayesian intervals; P–P plots; pull statistics | L9 |
| L11 | Ch 11 | reading an adjudication; evidence tags; pre-registration; confounded attribution | L10 |

**Rung-guard rule for the writer.** A chapter may not use a tool from a higher rung even
in an aside. Concretely: Ch 3 must talk about "how well the GW pins the distance" as a
*number with an error bar* and must not mention Fisher matrices (L6); Ch 5 must describe
w_G as "the probability the host is in the catalogue" and must not open β_G/D (that is a
Ch 9 object, once normalization is on the table). Where the temptation is strongest, the
book uses a **forward-reference token**: a small "⏭ Ch 6" chip the reader can click to
peek, which returns them to their place.

---

## Part 2 — The narrative arc as discovery

### 2.0 The spine

The book is one continuous experiment: **build an estimator for H₀, and keep trying to
break it.** Every chapter is a rung on that build, and the rung exists because the
previous rung visibly fails. The reader is never told "here is the correct formula"; they
are shown a posterior that is wrong, given a knob, and led to the term that fixes it.

The spine has a spine-of-its-own: the master equation, assembled one factor per chapter.

$$
p_i(h)\;=\;\underbrace{w_G(h)}_{\text{Ch }5}\;\underbrace{\mathcal L^{\rm cat}_i(h)}_{\text{Ch }3,4,7,8}\;+\;\underbrace{\bigl(1-w_G(h)\bigr)}_{\text{Ch }5}\;\underbrace{\mathcal L^{\rm comp}_i(h)}_{\text{Ch }5,9}
\qquad\Longleftrightarrow\qquad
p_i(h)=\frac{\beta_G(h)\,\mathcal L^{\rm cat}_i(h)+B^{\rm num}_i(h)}{D(h)},\quad D=\beta_G+\beta_{\bar G}
$$

with $w_G=\beta_G/D$ and $\mathcal L^{\rm comp}=B^{\rm num}/\beta_{\bar G}$
(`bayesian_statistics.py:3006-3009, 1042-1048`). The book's **hero graphic** is this
equation rendered with each factor greyed out until its chapter unlocks it. By Ch 8 it is
fully lit; Ch 11 puts three of the factors back into *amber* — the honest state.

### 2.1 Chapter partition — 12 chapters, each with its discovery statement

Format: **Title** — *the failure the previous chapter cannot answer* → *the mathematics
introduced as the resolution* → *what the reader leaves holding*.

---

**Ch 0 — Two Numbers That Should Be One**
*Failure:* nothing yet — but the ladder and the CMB disagree about H₀ at ~5σ, and every
rung of the local ladder is calibrated against another rung. There is no independent
ruler.
*Resolution:* state the target precisely — we want a distance measured by physics we
already trust, with a redshift attached to it, and we want to know exactly how wrong our
answer could be.
*Leaves holding:* the difference between *a number* and *a measurement with a stated
failure mode*; and the book's contract — we will build an estimator and spend most of the
book trying to break it.

---

**Ch 1 — A Ruler That Needs No Ladder**
*Failure:* the ladder needs calibration at every rung.
*Resolution:* general relativity gives the waveform amplitude ∝ 1/d_L with a known
constant. Introduce the EMRI: a 10 M☉ compact object spiralling into a ~10⁵–10⁷ M☉
massive black hole, ~10⁵ observable cycles in LISA's band, plunging inside the mission
window. Show that the *amplitude* carries d_L and the *phasing* carries the masses.
*Failure planted at the end:* the waveform gives d_L to a fraction of a percent — and
**nothing at all about z**. H₀ = cz/d_L needs both. A bright siren gets z from an EM
counterpart; an EMRI has none. The book's actual problem is now on the table.
*Leaves holding:* d_L is measured, z is not; "dark siren" is the name of that gap.

---

**Ch 2 — Bayes, Once and For All**
*Failure:* "I have one event with d_L and no z" is not yet an inference.
*Resolution:* Bayes over the single scalar $h=H_0/100$:
$p(h\mid\{d_i\})\propto p(h)\prod_i p(d_i\mid h)$, evaluated on a grid; log-additivity;
MAP vs posterior mean; the pull statistic $(\hat h - h_{\rm true})/\sigma_{\hat h}$;
**bias** defined as a property of the *estimator over realizations*, not of one run.
*Interactive failure:* stack real events and watch the posterior sharpen — except it
doesn't sharpen like $1/\sqrt N$. In the project's idealized baseline, **76 of 1588
events (4.8%) carry 100% of the constraint and 3 of them carry 46%**
(`IDEALIZED_BASELINE_READOUT.md:42-47`).
*Leaves holding:* N is not the number that matters; information is. And a vocabulary for
"wrong": bias, scatter, coverage, pull.

---

**Ch 3 — Which Galaxy?**
*Failure:* to turn d_L into H₀ you need a redshift, and you do not know which galaxy it
came from.
*Resolution:* marginalize the discrete latent. Sum over every catalogue galaxy $g$ inside
the GW localization volume, weighted by how likely that galaxy is to *host* an EMRI:
$\mathcal L^{\rm cat}=\sum_g w_g N_g / \sum_g w_g D_g$ — the **ratio of sums**, Gray et
al. (2020) Eqs. A.9/A.10, mapped symbol-by-symbol in `G2c`. The rate weight
$w_g = R_{\rm eff}(M_g)/(1+z_g)$ is *the same weight the simulator uses to draw hosts*
(`bayesian_statistics.py:2980-2990`) — the book flags this now and cashes it in Ch 9.
*The mechanism, made visible:* changing $h$ re-maps $d_L\to z$, which sweeps the
localization shell across a **different set of galaxies**. That sweep *is* how a
catalogue constrains H₀. Nothing else in the book matters if the reader misses this.
*Leaves holding:* the in-catalogue likelihood, and the fact that it is a ratio.

---

**Ch 4 — The Universe Only Shows You Its Loud Half**
*Failure:* run Ch 3's estimator on the real data and the posterior peaks **0.60** — the
bottom of the prior. Not near truth. Not even inside the plausible range.
*Resolution:* we conditioned on the wrong thing. The data is not "an EMRI happened"; it
is "an EMRI happened **and we detected it**". Conditioning on detection divides by the
selection integral $D(h)=\int p_{\rm det}\,(\text{rate})$. Introduce $p_{\rm det}$ as a
measured survival estimator built from the injection pool, and derive why the denominator
must be integrated over the **full volume**, not the local candidate window.
*The historical payoff, with real numbers:* the project shipped the local-window
denominator first. Replacing it with the full-volume $D(h)$ moved the MAP **0.60 → 0.73**
and the bias **−17.8% → 0.0%** (Phase 32; ledger #9). This is the book's cleanest
"without this the posterior is defective — watch" moment and it should be the reader's
first *earned* fix.
*Leaves holding:* selection is not a correction bolted on; it is what makes the
likelihood a likelihood. And: **the denominator is not a spectator** (a phrase the book
will need again, hard, in Ch 8).

---

**Ch 5 — The Galaxy You Cannot See**
*Failure:* Ch 4's estimator assumes the host is in the catalogue. For **95% of these
events it is not** — in the campaign venue only 76 of 1588 detections have an
in-catalogue host.
*Resolution:* a second, binary latent: is the host in the catalogue ($G$) or not
($\bar G$)? Law of total probability gives the two-branch mixture, and hence the master
equation. Introduce $\beta_G$, $\beta_{\bar G}$, $D=\beta_G+\beta_{\bar G}$,
$w_G=\beta_G/D$ = the posterior probability that *this detection's* host is catalogued,
and the completion term $\mathcal L^{\rm comp}=B^{\rm num}/\beta_{\bar G}$ built from the
population model where the catalogue is blind.
*Sub-derivation with a memorable failure:* the completion numerator must be marginalized
over the sky against an isotropic $1/4\pi$ prior. Evaluating it at the *peak* sky density
instead inflated $B^{\rm num}$ by **~5000×** and railed the posterior at the grid edge
(`G2a`; ledger #46). A factor of 5000 hiding inside a "surely it doesn't matter" step.
*Failure planted:* the two branches do not agree with each other. Split the events by
class and each class's summed log-posterior peaks somewhere different — **in-catalogue at
h = 0.86, dark at h = 0.64** (measured this session from
`real_r1/diagnostics/event_likelihoods.csv`). The headline number is where two runaways
cross. The reader is shown this and told: *hold that thought for six chapters.*
*Leaves holding:* the full master equation, and a splinter.

---

**Ch 6 — Opening the Black Box: What the Waveform Actually Measures**
*Failure:* the localization ball holds tens of thousands of galaxies. Ch 3–5 treated the
GW measurement as "an ellipsoid, somehow". To go further we need to know how big it
really is, what it is correlated with, and how much we should trust it.
*Resolution:* the Fisher matrix and the Cramér–Rao bound. The 14-parameter EMRI Fisher,
projected onto $(\phi,\theta,u)$ with $u=d_L/\hat d_L$, giving the 3D Gaussian the
likelihood actually evaluates (`bayesian_statistics.py:1052`, `982-1000`). Why sky
position and distance are *correlated* (the LISA antenna pattern varies across the sky).
Why the fractional distance coordinate is the natural one.
*Two catastrophic-factor stories, both real, both cheap to state:*
(i) the noise-weighted inner product was missing a $dt^2$ from the DFT↔continuous-FT
conversion — **every SNR 10× too small, every Fisher 100× too small, every CRB σ 10× too
large**, which capped the mock population at $z\le0.11$ instead of $\lesssim1.5$
(`G8`; ledger #51).
(ii) the sky/frame story: a Cramér–Rao covariance stamped in the wrong frame moved the
MAP **0.860 → 0.730** by itself and raised host recovery 31→38 of 60 (ledger #12).
*Leaves holding:* where the numbers in the ellipsoid come from, and a visceral sense that
a single misplaced Jacobian or sampling factor is not a rounding error — it is the
difference between a measurement and noise.

---

**Ch 7 — A Redshift Is Not a Number**
*Failure:* the estimator so far plugs each catalogue galaxy's redshift in as if it were
exact. It is not: GLADE+ redshifts are mostly photometric — **spectroscopic redshifts are
0.56% of GLADE+** (ledger #42) — and the campaign's information-carrying hosts have
**median σ_z/z ≈ 49%** (`IDEALIZED_BASELINE_READOUT.md:50-52`). Plug in the central value
and the posterior is biased. **Watch it happen as you turn σ_z up.**
*Resolution:* a likelihood is not a posterior. $p(z_g\mid z)$ read as a function of $z$
needs a prior before it can weight anything, and the right prior is the *rate*:
$w_{\rm pop}\propto (dV_c/dz)/(1+z)$. This is the `volume_deconv` kernel, derived in
`G2b`, with per-galaxy renormalization $Z_g$ and the "$dV_c$ counted once" symmetry
between numerator and denominator.
*The law, with the project's own measured points:* the bare-Gaussian kernel produces an
Eddington-type bias $\Delta h = -C\sigma_z^2$ with $C_{\rm meas}\approx17$–$20$, measured
at **−0.0016 / −0.0064 / −0.023 / −0.046** for σ_z = 0.005 / 0.015 / 0.035 / 0.050;
`volume_deconv` takes the bias **−0.024 → −0.002** and coverage from ≈0% to nominal
(`G2b:229-237, 413-436`; ledger #47). The reader turns a σ_z dial and lands on those four
points.
*The twist that must be told here, not hidden:* the same deconvolution has a failure mode
of its own. Deconvolving a wide photo-z against a **monotonically rising** volume prior
with **no selection turnover** pushes the host-z estimate *up*. The corrected law measured
on this code is
$h_{\rm eff}/h_{\rm true} = \tfrac12\bigl[1+\sqrt{1+12(\sigma_z/z)^2}\,\bigr]$, i.e.
$\to 3(\sigma_z/z)^2$, with **rail threshold σ_z/z > 0.256** — and the campaign's
in-catalogue hosts sit at σ_z/z = 0.25–0.49 (claim **C7**, adjudicated FINDING (MEASURED),
2026-07-30). The fix and its own failure mode are the same knob. The book says so in the
chapter that introduces the fix.
*Leaves holding:* the σ_z² law, the volume prior, and the first live amber flag.

---

**Ch 8 — A Second Handle: the Mass Channel**
*Failure — of the opposite kind:* here the motivation is **enhancement**, not defect. The
GW phasing measures the *redshifted* mass $M_z = M(1+z)$ to $\sigma_{M_z}/M_z\approx10^{-4}$.
If a catalogue galaxy tells you its black-hole mass $M_g$, then $M_z/M_g - 1$ is a second,
independent estimate of $z$ — a second handle on the same event. It should sharpen
everything.
*Resolution:* the 2D (z, M) kernel — truncated lognormal on the catalogue's
Reines & Volonteri stellar-mass proxy (σ_lnM ≈ 1.28 ≈ a factor 3.6), the analytic $M_z$
fraction marginal in the numerator, counted-once-in-M in the denominator
(`mass_marginal_2d_kernel.md`, RATIFY-M1..M7), plus the Eddington-in-M rate prior
(`G2d`).
*It works, mechanically:* the mass window rejects **97–99% of impostor hosts**. At h=0.73,
**64.7% of dark events have $\mathcal L^{\rm cat}_{\rm 2D}$ exactly zero** (vs 32.5% in
1D), survivors suppressed by a median factor **7.8×10⁻³**, and the rejection is strictly
**one-sided** — 193 low-side vs 1 high-side — because σ_Mz/M_z ≈ 10⁻⁴ makes the window's
upper leg vacuous (C4-obs, LOCAL VERIFIED, replicated 10/10 runs).
*And the posterior gets worse.* Campaign #53: 1D pooled mean 0.7205, **2D mean bias
+0.077 with 10/10 runs pulling > 2 (+4.04)** (`REALISTIC_READOUT.md`). Rejecting impostors
is exactly what the 2D channel is *for*, and it moved the answer away from truth.
*The amended mechanism, stated as the project states it:* deletion is **not** the
mechanism — the 487 events zeroed at every h carry only **+0.24 nats = 1.5%** of the
channel difference. **98.5% is carried by the 534 survivors, whose catalogue mixture
weight is de-weighted 0.0354 → 0.0061 (a factor 5.8)**; and the up-pull that then wins is
carried by the **(1−w_G) prefactor**, not by $\mathcal L^{\rm comp}$ — which for dark
events pulls *down* (only 39.1% positive tilts; C10). The book must not write "the
completion term pulls up"; C10 exists precisely to retire that sentence.
*The well-posedness bomb (C8):* the 2D channel is **reparametrization-dependent**.
Rescaling the mass coordinate by a constant C walks the MAP **0.81329 → 0.78107 → 0.74440
→ rails at 0.600** for C = 1 / 0.3 / 0.1 / ≤0.01, at **+0.031 per e-fold**, while the 1D
channel is **bitwise invariant**. Cause, relocated by adjudication: the two numerator legs
carry different mass dimensions — the catalogue leg carries one mass density, the
completion leg carries none — and the code silently hard-wires the measure to
$dM_z/M_{z,{\rm det},i}$, *the event's own measured mass*, which spans a factor 12 across
events. A number that moves when you change your unit is not a measurement.
*Leaves holding:* the master equation fully lit — and two of its factors flashing amber.

---

**Ch 9 — Building a Universe to Break Your Estimator**
*Failure:* every claim so far compared an estimator to "truth". Where did truth come
from? And how would you notice if your simulator and your estimator disagreed about the
same universe?
*Resolution:* the forward model. The Barausse (2012) M1 EMRI population with
$\Omega_m = 0.2726$ chosen *to match the population*, not the CMB (a documented design
choice, G11, not a bug — ledger #53). The plunge-window convention: an "event" is an EMRI
that plunges inside the mission span, $t_{\rm plunge}\sim U[0,T]$, $T=4.5$ yr, with $p_0$
solved by root-finding on the same PN5 trajectory that generates the waveform
(`plunge_window_initial_conditions.md`, ratified 2026-07-28). The injection pool and the
survival-estimator $p_{\rm det}$. The realistic host-observation model (R1–R9) that
replaces exact host redshifts and masses with *realized* noise.
*The deep idea of the chapter — normalization modes.* $\mathcal L^{\rm cat}$ can be
normalized in more than one internally consistent way (`generator_marginal` vs
`absolute_marginal`), and the choice changes what $w_G$ *means*. The same run reports
$w_G(0.73)=0.1215$ under one mode and $0.0497$ under the other — **two different
estimands, neither corrupt**, and comparing across them silently produced a phantom "45%
discrepancy" that cost the project real time.
*The live defect (C9), stated honestly:* the delivered $w_G(0.73)=0.1215037$ against the
realized detected in-catalogue rate **164/3135 = 0.05231**, binomial **z = −11.86**. The
whole gap localizes to the catalogue's *relative detection efficiency*: $\beta_G$ weights
by the pool-marginal $p_{\rm det}$, but Malmquist-selected catalogue hosts carry heavier
M–σ black holes. Two independent suppression measures agree to 0.2σ (0.39248 vs
0.3991 ± 0.0312) and the mass-aware $w_G = 0.05149$ lands at z = +0.21. The generator and
the estimator are describing different universes — **by a factor 2.3**.
*Leaves holding:* the idea that "is my estimator biased?" is meaningless without "biased
relative to what generator?", and the first taste of confounded attribution.

---

**Ch 10 — Is It Calibrated?**
*Failure:* Ch 9's idealized baseline recovers truth beautifully — MAP **0.72990**,
σ_h = 0.00030, **−0.24σ** at seed 61000; **0.72990, −0.36σ** at 62000. Is the estimator
therefore good?
*Resolution:* no, and the chapter says why in three moves.
(i) **A single MAP hitting truth is not calibration.** Introduce coverage, the P–P plot,
and the pull distribution over many synthetic universes (`validation/pp_coverage.py`).
(ii) **Where the closure came from matters.** The redteam measured that **133 golden
events carry ~100% of the curvature** — so the closure test validates *host association*,
not the selection machinery (ledger #85, R2). A test can pass for the wrong reason.
(iii) **The idealization ledger.** Recovery-on-truth in #51 is guaranteed by construction:
host z and M are used point-to-point, no measurement noise realized. The book quotes the
readout's own caveat rather than paraphrasing it.
*The calibrated result the harness does give:* the completion leg's deep-venue
mis-calibration is **+0.0008…+0.0097** at the campaign's completeness fraction — real,
monotone, sign-stable, and **6–16× too small to own the +0.077** (C11). A quantitative
exoneration, which is a different and better thing than a hand-wave.
*Leaves holding:* coverage as the real currency; and the discipline of asking what a
passing test actually tested.

---

**Ch 11 — The State of the Art, Honestly**
*Failure:* everything.
*Resolution:* none yet — and that is the chapter.
This chapter presents, in the project's own adjudicated language:
- **C5** — 58% (44/76) of in-catalogue hosts rail at the h = 0.86 prior edge (5.3%
  idealized), replicating 54–67% in 10/10 runs against a 2.4% flat-surface expectation.
  The rail is a **clipped real runaway**, not an edge artifact: railed profiles are
  genuinely concave and extrapolated vertices land at $h_{\rm eff}=0.93$–$1.05$.
  **With the binding fair-framing amendment**: *per event the rail is cosmetic*
  (0.072–0.134 nats = 0.30–0.47 σ_event), but the *class-summed* displacement is
  **+3.4 to +6.1 σ_class** — coherent, not noise. Both halves of that sentence are
  mandatory; the book may not quote one without the other.
- **C7** — the host-z numerator kernel omits selection. FINDING (MEASURED). And it
  **collides with the ratified G2b derivation**, which confirmed exactly that weight,
  without $p_{\rm det}$, as uniquely consistent — protected by a binding regression gate.
  Any fix must *explicitly supersede* G2b. The book shows the collision rather than
  picking a side.
- **C8** — reparametrization dependence: a well-posedness defect, independent of any bias.
- **C9** — the $w_G$ mis-calibration, and its extension to the dark side (KS D = 0.0863,
  p = 1.08×10⁻¹⁹ against the pool's own coded integrand).
- **C6** — and the honest capstone: **attribution is confounded.** Campaign #51 → #53
  changed **three variables at once** (catalogue scatter, host-z kernel, normalization
  mode) and *no run anywhere varies the estimator at fixed catalogue*. The decisive
  control — pre-registered 2×2 cell B — was in flight at the time of writing. The book
  reproduces the **dated pre-readout prediction** ("B ≈ C in both channels") so the reader
  can see what pre-registration looks like from the inside.
- **The direction, not the verdict:** all three measured internal inconsistencies (C7, C8,
  C9) sit on the **completion/prefactor/kernel** side; none convicts the catalogue leg;
  and the mass de-weighting is the 2D channel's *intended function*. Evidence-weighted,
  not decided.
*The payoff of Ch 5's splinter:* the Two Runaways figure returns, now with the leverage
measurement — a ±1/√N Poisson reweight of the two classes moves the combined MAP by up to
**0.025** (vs **0.0000** idealized), and $dh^*/d\varepsilon$ leverage is **1500–2400×** the
idealized value. The reader who held the thought since Ch 5 gets to *feel* why "0.72" was
never the point.
*Leaves holding:* what an honest open problem looks like written down, and why the last
chapter of a real project is a list of live threads rather than a result.

---

### 2.2 The recurring beats, and where each pays off best

Seven beats. Each chapter uses 4–6 of them; none uses all seven (that would be a
template, and readers detect templates).

**B1 — Cold Open (every chapter, 1st).**
A live, broken posterior before a single line of math. The reader sees the failure in the
first screen. *Best payoffs:* Ch 4 (MAP sits at 0.60 — nothing else in the book is this
stark), Ch 7 (drag σ_z, watch the peak slide), Ch 8 (the reader has just been *promised*
an improvement, and the mean bias goes to +0.077 — the book's best cold open because it
inverts the reader's expectation rather than confirming a fear).

**B2 — Predict-Then-Reveal (2–3 per chapter).**
The reader commits before seeing. Implementation: a slider or a 3-way choice that must be
set before the "Reveal" control unlocks; the reader's guess is then drawn as a ghost line
alongside the truth and **persists for the rest of the book** in their margin. This is not
a quiz — it is a record of the reader's own priors being updated, and re-surfacing their
Ch 3 guess in Ch 11 is one of the strongest devices available.
*Best payoffs:* Ch 4 ("where will the MAP go when I delete the denominator?" — most
readers guess "a bit low"; the answer is *the edge of the prior*), Ch 8 ("the mass channel
adds information — will the bias go up, down, or stay?"), Ch 10 ("this estimator recovered
truth at −0.24σ. Is it calibrated? y/n" — nearly everyone says yes).

**B3 — Break-It-Yourself sandbox (1 per chapter from Ch 4 on).**
A sandbox in which the reader can *disable* the term the chapter just introduced, or push
it past its validity, and watch the estimator degrade. The rule: the sandbox always uses
**real campaign data**, never a toy, so the failure the reader produces is the failure the
project measured.
*Best payoffs:* Ch 4 (denominator on/off/local), Ch 7 (σ_z dial + "selection turnover"
toggle — the reader breaks the fix with the fix's own knob), Ch 8 (the Reparametrization
Walk — the reader *changes their mind about units* and the answer moves).

**B4 — EMRI-889, the running worked example (every chapter).**
**Event 889 of seed 61000, realization r1.** It is the ideal protagonist and the data is
already local:
- M = 7.25×10⁵ M☉, μ = 10 M☉, **d_L = 88.9 Mpc**, **SNR = 1425** (the loudest of 1590),
  fractional distance precision **σ_dL/dL = 8.0×10⁻⁵**, host galaxy index 859360, in
  catalogue.
- It is one of the **3 golden events that carry 46% of the idealized constraint**.
- Its 1D likelihood is a clean, beautiful peak: $\mathcal L^{\rm cat}$ rises 0.39 → 8.79 →
  12.49 and falls back to 1.48 across the h-grid, and the combined 1D likelihood peaks at
  **h ≈ 0.745–0.750**.
- Its 2D likelihood is **nearly flat** — $\mathcal L^{\rm cat}_{\rm 2D}$ crawls 0.47 →
  1.31 → 1.70 monotonically — and the combined 2D curve peaks at **h ≈ 0.79–0.80**. The
  mass channel *destroyed this event's information and pushed it high*.
- It is also **the single high-leverage in-cat event** whose channel difference swings
  +1.98 → −2.04 → −3.30 across noise realizations r1/r2/r3 of the same class, and which
  alone flips the sign of the in-cat component in run r3.

So one event carries: the loud clean measurement (Ch 1, 2, 6), the host ambiguity (Ch 3),
the golden-event information concentration (Ch 2, 10), the photo-z problem (Ch 7), the 2D
catastrophe (Ch 8), and the realization-to-realization fragility (Ch 11). **It is the
whole book in one object.**
*Counterpart:* **dark event 606** (SNR 43, d_L 1.17 Gpc) as the contrast — 1D catalogue
leg nearly flat at ≈0.088, 2D leg suppressed 80× to ≈0.0012 and *decreasing* in h, while
$\mathcal L^{\rm comp}$ rises monotonically 0.0149 → 0.0238 and the combined 2D curve
rises all the way to the top of the grid. Every mechanism in Ch 8 is legible in these two
events side by side.
*Presentation:* a persistent, collapsible **Event Dossier** in the page chrome. It gains a
row each chapter — "what we knew about 889 after Ch N" — so the reader can watch a single
event acquire a redshift, a host, a weight, an error bar, and finally a warning label.

**B5 — Defect Museum interlude (5 placements).**
Short, self-contained, between chapters. Each exhibit: *the hypothesis · the
pre-registered prediction · the decisive test · the measured outcome · what it cost · the
transferable lesson.* Drawn from the 98-row ledger. Placements chosen so each interlude
inoculates against a mistake the *next* chapter invites:

| After | Exhibit | Ledger | Lesson it inoculates |
|---|---|---|---|
| Ch 4 | **The p_det anchor** — raise the d_L→0 anchor by +12%, predicted to move the MAP; moved it **zero grid steps** | #17 | "the layer you can see is not the layer that matters" |
| Ch 5 | **Information starvation** — the project formally concluded in-catalogue photo-z dark sirens are information-starved, then **overturned it**: starvation is "a property of prior-INCONSISTENT estimators, not of the data" | #41 → #52 | a negative result about *your estimator* is not a negative result about *nature* |
| Ch 7 | **`volume_trunc`** — a pre-registered fix that made the bias **worse by ~4×** (1D 0.745 → 0.800), for **two** independent reasons at once, one of them `fixed_quad(n=50)` **aliasing the GW peak to 0.0000** where the exact value was 0.24–0.65 | #70 | a wrong answer can have two causes, and quadrature is physics |
| Ch 8 | **`mass_trunc`** — a mass-kernel fix **confirmed in isolation** (+0.016…+0.02, right sign) and then **exonerated in the full pipeline** (Δ2D = +0.0029, wrong sign; Δ1D = 0.0000 exact), because the same prior enters $D_g$ and cancels: *"the selection denominator is not a spectator"* | #71 → #72 | isolation tests can be right and irrelevant |
| Ch 10 | **The H0-independent estimator** — a 10-way injection scan found the production MAP was **0.86 for every injected truth from 0.63 to 0.77**, while `catalog_only` tracked truth exactly | #49a | an estimator can be precise, stable, reproducible, and carry **zero information about the parameter** |

A sixth exhibit, **the archaeology** (git-bisect of stored posteriors located the rail's
birth at ~2026-04-24, when the Gray completeness machinery was switched on, against an
unbiased 2026-04-09 baseline; ledger #49b), sits in the Museum annex as the "how do you
even find out when it broke" methods exhibit.

**B6 — Misconception traps (1–2 per chapter, in-line).**
Stated as a plausible sentence the reader is *likely to be thinking*, in the reader's own
voice, then dismantled with a measurement. Full list in Part 3.

**B7 — Self-check block + Transfer question (every chapter, last).**
3–6 questions, hidden answers, the last one always a **transfer** question whose answer is
the next chapter's opening failure. Full set in Part 3.

### 2.3 Three book-wide instruments (the structural bets)

**BW1 — The Bias Ledger Rail.** A thin persistent rail along the page edge showing *the
current chapter's estimator* and its **measured** bias in h, from the project's own
history: Ch 3 → not defined yet; Ch 4 → **−17.8% → 0.0%** as the reader flips the
denominator; Ch 7 → the σ_z ladder −0.0016…−0.046 and its repair to −0.002; Ch 8 →
**+0.077** appears in the 2D channel; Ch 11 → the honest amber state. Clicking any rung
opens the artifact. **This is the book's single most important structural device**: it
makes "the estimator is a thing under construction with a live scorecard" a felt property
of reading rather than a claim.

**BW2 — The Symbol Passport.** Hovering any symbol anywhere gives: definition, **units**,
the code site (`file:line`), the derivation that ratified it, and its current status
badge. Clicking pins it to a personal glossary. This is how a book with ~40 symbols stays
readable for Mara and simultaneously serves Tomas's need for exact correspondence.

**BW3 — "Has this been tried?"** A search box over the 98-row ledger, available from every
Break-It-Yourself sandbox. When the reader's sandbox configuration matches a historical
hypothesis, the box **volunteers** the verdict: *"You just built hypothesis #61. It was
tried on 2026-07-11. It relocated the tilt to the host branch, +94…+455 nats, 12/12
fail."* Turning a research ledger into a live antagonist is, as far as we can tell, new —
and it teaches the single most valuable research habit in the book: **check whether your
idea is already dead.**

---

## Part 3 — Question design

**Design rules.** (a) No recall questions — every question is "what happens if", "why can
this not be dropped when", an order-of-magnitude estimate, or a transfer. (b) Answers are
hidden behind a disclosure and are **one paragraph**, always naming the mechanism, not
just the outcome. (c) Every chapter's last question is the **transfer** question. (d) Each
chapter carries 1–2 **misconception traps** written in the reader's voice. (e) Where a
number appears in an answer, it is a real project measurement and the answer says where
it came from.

---

### Ch 0 — Two Numbers That Should Be One

**Q0.1** The two H₀ values differ by about 8%. Order of magnitude: if a single new method
is to *arbitrate*, roughly what fractional precision does it need, and why is "better than
8%" not the right answer?
> **Answer.** It needs to be several times better than the gap — of order 1–2% — because
> arbitration requires distinguishing two hypotheses that are 8% apart *with confidence*,
> and a method whose own 1σ is 8% would place both values inside its own interval. The
> deeper point is that precision is necessary but not sufficient: a 1% method with a 3%
> unknown systematic arbitrates nothing. This is why the rest of the book spends far more
> effort on *bias* than on σ.

**Q0.2** Why does it matter that the new method's systematics be *uncorrelated* with the
ladder's, rather than merely small?
> **Answer.** Because a discrepancy between two methods is only informative if their
> errors are independent. If a dark-siren H₀ inherited the same Cepheid or SN calibration
> as the ladder, agreement would be circular and disagreement uninterpretable. GW sirens
> are attractive precisely because the distance comes from the waveform amplitude and
> general relativity, sharing no calibration rung with the EM ladder — the systematics we
> spend this book on (catalogue completeness, photo-z, selection) are *different*
> systematics, which is the whole point.

**Q0.3 (transfer)** GR gives an absolute distance from the waveform. Name the one thing
you still need before that distance becomes a Hubble constant, and guess how hard it is to
get for a source with no light.
> **Answer.** A redshift. H₀ ≈ cz/d_L requires both, and the waveform supplies only d_L.
> For a source with an electromagnetic counterpart the redshift is a spectrum away; for a
> source with none — which is every EMRI — you must infer it, and the entire rest of this
> book is the cost of that inference.

**Trap 0.A — "Sirens are better because they're direct."**
> Directness buys you freedom from *calibration* systematics, not freedom from
> systematics. This book will show a fully "direct" estimator sitting at h = 0.60, at
> 0.86, and reparametrization-dependent — none of which is a calibration problem.

---

### Ch 1 — A Ruler That Needs No Ladder

**Q1.1** The waveform amplitude scales as 1/d_L. Why does that not immediately mean the
distance is measured as well as the amplitude?
> **Answer.** Because the observed amplitude is a product of distance, source masses, and
> the geometric factors of inclination and sky position folded through the detector's
> antenna pattern. Amplitude alone is degenerate; the distance is recovered only because
> the *phasing* independently fixes the masses and the modulation of the signal across the
> mission fixes the geometry. That is why an EMRI's ~10⁵ cycles are so valuable — they
> break the degeneracies that make a short binary's distance poor.

**Q1.2** EMRI-889 has SNR 1425 and a fractional distance precision of 8.0×10⁻⁵. Estimate
what fractional H₀ precision that *single* event would give if its host redshift were
known exactly, and say why the book will nevertheless not get anywhere near that.
> **Answer.** With z exact, σ_H0/H0 ≈ σ_dL/dL ≈ 0.008% — an absurdly good single-event
> measurement, and indeed the idealized campaign's total σ_H0/H0 ≈ 0.044% is roughly
> 0.38%/√76, of the same character. The book will not get near it because (i) the host
> redshift is *not* known exactly — it is a photometric redshift with median σ_z/z ≈ 49%,
> which dominates everything, and (ii) for 95% of events there is no identified host at
> all. The GW measurement is not the bottleneck; the galaxy catalogue is.

**Q1.3** Why is "an event" defined in this pipeline as an EMRI that **plunges inside the
mission span**, rather than one that is simply loud enough at some moment?
> **Answer.** Because the SNR of an EMRI accumulates overwhelmingly in the final,
> strongest-field cycles near plunge, so the detectable population is essentially the
> plunging population; and because the rate model the events are drawn from (Babak et al.
> 2017 M1) is itself a *plunge rate*. Defining events any other way would put the
> generator and the population model in disagreement — the first instance of a theme that
> becomes Ch 9.

**Q1.4 (transfer)** You have d_L to 0.008% and no redshift. Sketch, in one sentence, the
cheapest possible way to get a redshift, and name the assumption it smuggles in.
> **Answer.** Look up which galaxies lie inside the GW localization volume and use their
> redshifts — which smuggles in the assumptions that the host is (a) *in* the catalogue,
> (b) identifiable among the candidates, and (c) at its catalogued redshift. Chapters 3, 5
> and 7 are the successive discoveries that all three are false.

**Trap 1.A — "An EMRI is just a long binary."**
> The length is the physics. ~10⁵ cycles deep in the strong field is why $M_z$ is measured
> to ~10⁻⁴ — a precision that becomes, in Ch 8, both the best idea and the worst problem
> in the book.

---

### Ch 2 — Bayes, Once and For All

**Q2.1** You add a second event. The combined posterior is the product of the two
likelihoods. Under what circumstance does a second event make the posterior *worse* rather
than merely wider?
> **Answer.** When the second event's likelihood is biased in a way the first is not,
> because products of likelihoods are unforgiving: a coherent, same-signed tilt shared
> across many events accumulates linearly in log-space while the statistical width shrinks
> as 1/√N, so the *pull* grows as √N. This is precisely the failure mode the book ends
> with: campaign #53's 2D channel produces 10 runs all pulling in the same direction at
> +4.04σ. More data made the wrongness more significant, not less.

**Q2.2** In the idealized baseline, 76 of 1588 events carry 100% of the constraint and 3
carry 46%. What does that do to your intuition about the estimator's robustness?
> **Answer.** It destroys the comfort of large N. With effectively ~3–76 informative
> events, the posterior inherits the idiosyncrasies of a handful of galaxies: their
> redshift errors, their catalogue masses, their sky positions. A single event's
> misbehaviour can then move the answer, which is exactly what the project measured for
> event 889 — its channel contribution swings +1.98 → −2.04 → −3.30 across three noise
> realizations of the *same* event list, flipping the sign of its whole class's
> contribution in one of them.

**Q2.3** Distinguish, with a one-line operational test each: a *biased* estimator, a
*noisy* estimator, and a *mis-calibrated* estimator.
> **Answer.** Biased: repeat the experiment many times with the same truth and the mean of
> the estimates misses truth. Noisy: the estimates scatter widely but their mean is right.
> Mis-calibrated: the estimates may even be unbiased, but the *stated* uncertainty is
> wrong, so the 68% intervals do not contain truth 68% of the time. The three are
> independent, and Ch 10 exists because a project can be congratulating itself on the
> first while failing the third.

**Q2.4** Why does this pipeline evaluate the posterior on an h-**grid** rather than
sampling it?
> **Answer.** Because every event's likelihood requires an expensive integral over
> candidate hosts, and h enters only through the d_L↔z mapping, so the same host set can
> be re-evaluated at a fixed list of h values and cached — h is one scalar, not a
> high-dimensional space needing MCMC. The cost is that the grid must resolve the peak:
> the idealized posterior's σ_h = 0.00030 is ~15× narrower than the production grid's
> 0.005 step, which is why the un-zoomed posterior looked like a delta spike and why the
> project needed a zoom-grid hook.

**Q2.5 (transfer)** You have a posterior machine and a distance. To get a likelihood
$p(d_i\mid h)$ you must connect d_L to h. Write down the missing ingredient and predict
how many candidate values of it a typical event will offer.
> **Answer.** The host redshift; and typically *tens of thousands* of candidates, because
> the localization volume of even a well-localized EMRI contains that many catalogued
> galaxies. The likelihood must therefore marginalize over "which galaxy", which is Ch 3
> — and the fact that the answer is usually "none of them" is Ch 5.

**Trap 2.A — "MAP = truth means the estimator is right."**
> Ch 10 exhibit #49a: a production estimator returned MAP = 0.86 for *every* injected
> truth from 0.63 to 0.77. It was stable, reproducible, and carried literally zero
> information about H₀. Hitting truth once is one bit of evidence.

**Trap 2.B — "More events always help."**
> Only if they are unbiased. See Q2.1: 10/10 runs at +4.04σ is what "more events" does to
> a coherently tilted estimator.

---

### Ch 3 — Which Galaxy?

**Q3.1** The likelihood sums over candidate hosts weighted by $w_g$. What happens to the
inference if you set all $w_g$ equal?
> **Answer.** You assert that every galaxy in the ball is equally likely to host an EMRI,
> which is false: the EMRI rate per massive black hole rises with the black-hole mass
> (d log R_eff/d log M ≈ −0.43 per dex in the *mass-function-weighted* sense used here),
> and the number of hosts per unit volume is not the number of *events* per unit volume.
> Equal weights therefore mis-state which redshifts the event could have come from, and —
> more subtly — put the estimator into disagreement with the simulator that drew the host
> using the rate weight, which is the Ch 9 failure mode in miniature.

**Q3.2** As you increase h, the same measured d_L maps to a larger z, so the candidate
shell sweeps outward. Why is that sweep the *entire* mechanism by which a catalogue
constrains H₀?
> **Answer.** Because the GW data never sees a redshift; it sees a distance. The only way
> h can be preferred or disfavoured is if some values of h place the shell on top of real
> galaxies and others place it in empty space. The likelihood is literally a measure of
> "how much galaxy is where this h says the source must be". If the catalogue were uniform
> and structureless, or if the shell were much wider than the structure, the sweep would
> carry no information — which is the quantitative content of Ch 7's photo-z problem.

**Q3.3** The in-catalogue likelihood is a **ratio of sums**, $\sum_g w_g N_g / \sum_g w_g D_g$,
not a sum of ratios $\sum_g w_g (N_g/D_g)$. Why can that distinction not be waved away?
> **Answer.** Because the two differ whenever the per-galaxy denominators $D_g$ vary across
> the candidate set — which they do, since $D_g$ depends on each galaxy's own redshift and
> detectability. The mean-of-ratios form over-weights galaxies that are individually hard
> to detect. The project shipped the wrong form and fixing it to Gray's A.9/A.10 ratio-of-
> sums **halved** the 1D bias, 0.750 → 0.740 (ledger #26) — an "obviously equivalent"
> rearrangement worth 0.010 in h.

**Q3.4** Order of magnitude: a well-localized EMRI at z ≈ 0.1 with a ~10 deg² sky area and
0.01% distance precision. Roughly how many catalogued galaxies are in the ball, and what
does that imply about the strength of a single event?
> **Answer.** Tens of thousands. Even with a razor-thin distance shell, the *angular*
> uncertainty dominates the volume, and the catalogue's density at low z is high. A single
> event therefore rarely identifies a host; it supplies a weak preference over a large
> candidate set, and the constraint comes from stacking. The exception is the golden event
> — for EMRI-889 the shell is thin enough and the sky area small enough that the sum is
> dominated by very few galaxies, which is exactly why 3 events carry 46% of the
> information.

**Q3.5 (transfer)** Run this estimator on the real event set and the posterior peaks at
h = 0.60, the bottom of the prior. Before reading on: name one thing this likelihood
conditions on incorrectly.
> **Answer.** It conditions on "an EMRI occurred at this galaxy", when the data is "an
> EMRI occurred at this galaxy **and we detected it**". Detection preferentially selects
> loud, nearby sources, so the likelihood as written rewards h values that put sources
> where detection is easy. Correcting that requires dividing by the probability of
> detection integrated over everything the source could have been — Ch 4.

**Trap 3.A — "The right host is the closest galaxy to the best-fit sky position."**
> The likelihood is a *weighted sum*, not a nearest-neighbour lookup, and treating it as
> one throws away the 3D correlation structure between sky position and distance that the
> Fisher matrix (Ch 6) supplies. In the campaign venue, the events whose likelihood is
> dominated by one galaxy are 3 out of 1588.

**Trap 3.B — "If we can't identify the host, the event is useless."**
> Ledger #41 → #52: the project formally concluded exactly this ("information starvation")
> and then **overturned it**. The 1512 dark events are not useless; they are handled by a
> different branch (Ch 5), and in the 2D channel they end up carrying 84%+ of the channel
> difference.

---

### Ch 4 — The Universe Only Shows You Its Loud Half

**Q4.1** You delete the selection denominator entirely. Predict the direction of the bias
in h *before* reasoning about the formula, then check yourself.
> **Answer.** The undenominated likelihood over-rewards configurations where detection is
> easy — nearby sources — so it prefers h values that make the observed distances
> correspond to *low* redshift, i.e. it rails **low**. The project measured exactly this:
> before the full-volume denominator, the MAP sat at 0.60, the bottom of the prior, with a
> bias of −17.8%; installing the correct D(h) moved it to 0.73 and the bias to 0.0%
> (ledger #9).

**Q4.2** Why must the denominator be integrated over the **full volume** rather than over
the same local candidate window as the numerator? The local version looks more
"consistent".
> **Answer.** Because the denominator answers a different question. The numerator asks
> "given this h, how well does this data match a source at these candidate hosts?"; the
> denominator asks "given this h, what fraction of *all* sources that could have occurred
> would have been detected at all?" — a question about the whole universe the population
> model describes, not about this event's neighbourhood. Restricting it to the window
> makes the normalization event-dependent in a way that does not cancel, and the project
> measured that this single mis-scoping was the dominant bias of its era.

**Q4.3** $p_{\rm det}$ is estimated from a finite pool of injections. Name two distinct
ways a finite pool can bias the *inference* rather than merely add noise.
> **Answer.** (i) Extrapolation policy: filling $p_{\rm det}$ with 0 outside the grid
> silently deletes real support, which the project had to fix explicitly (ledger #8, −9.2%
> → −6.9%); (ii) support mismatch — if the pool is thin where the catalogue's candidates
> actually live, the estimator's effective sample size collapses in exactly the cells that
> matter, which is why the project's ratified campaign design imposes a **minimum ESS per
> node** rather than trusting a large total pool. Both are systematic, not stochastic.

**Q4.4** The selection integral depends on h. Why does that dependence *not* simply cancel
between numerator and denominator?
> **Answer.** Because they are integrals of different things over different supports: the
> numerator carries the GW measurement's own likelihood evaluated at the candidate hosts,
> the denominator carries the population rate times detectability over the whole volume.
> Their h-dependences are genuinely different functions, and the ratio's h-dependence is
> the selection correction. When a piece of it *does* cancel, that is a theorem worth
> proving, not an assumption — the project's G1 gate checked one such cancellation
> ($\Sigma_{\rm glob} \equiv \bar n\,\beta_G$) and found a **−17.2% end-to-end residual**,
> which is now part of claim C9.

**Q4.5 (transfer)** Your fixed estimator now recovers truth in a mock where every host is
in the catalogue. In the real venue, only 76 of 1588 events have a catalogued host. What
does your likelihood currently say about the other 1512?
> **Answer.** Nothing coherent — as written it either returns zero (if the candidate list
> is empty) or the likelihood of the wrong galaxies. The project actually shipped the
> "returns zero and the event is silently dropped" version, which discarded **58% of
> events** on a deep venue before it was found and fixed (ledger #54). The correct answer
> requires a second branch for uncatalogued hosts: Ch 5.

**Trap 4.A — "Selection effects are a small correction."**
> They were worth 17.8% in h — larger than the entire Hubble tension the book opened with.

**Trap 4.B — "p_det is a detail of the simulation, not of the physics."**
> $p_{\rm det}$ appears in the denominator of every event's likelihood and its h-dependence
> is a direct multiplicative tilt on the posterior. Ledger #33: fixing a tail overshoot in
> the $p_{\rm det}$ estimator moved the MAP 0.760 → 0.750.

---

### Ch 5 — The Galaxy You Cannot See

**Q5.1** $w_G$ is "the probability that this detection's host is in the catalogue". Why is
it a function of **h**, and which way does it go as h increases?
> **Answer.** Because h sets the distance–redshift mapping, and therefore how far out in
> redshift a source at a given measured d_L must be. Larger h pushes sources to higher z,
> where the catalogue is more incomplete, so $w_G$ **falls** with h. This is measured
> directly in the campaign diagnostics: $w_G$ runs 0.1625 → 0.0947 across h = 0.60 → 0.86,
> monotonically. That slope is not a detail — Ch 8 shows the $(1-w_G)$ prefactor's tilt is
> worth +31.55 nats across a 0.08-wide h window, larger than any other single term.

**Q5.2** The completion term is built from the population model in the region where the
catalogue is blind. Why can it not simply be "the catalogue term with the missing galaxies
filled in"?
> **Answer.** Because you do not know where the missing galaxies are — that is what
> "missing" means. The completion branch replaces the *discrete sum over known positions*
> with a *smooth integral over the population's expected distribution*, marginalized over
> the sky against an isotropic prior. It is a genuinely different mathematical object, and
> the switch from discrete to smooth is where a factor of 5000 hid: evaluating the sky
> marginal at the peak of the density instead of integrating $1/4\pi$ inflated $B^{\rm num}$
> ~5000× and railed the posterior at the grid edge (G2a, ledger #46).

**Q5.3** Sanity limits. What must the master equation reduce to as (a) the catalogue
becomes complete, (b) the catalogue becomes empty?
> **Answer.** (a) $w_G\to1$, the completion branch switches off, and $p_i \to
> \mathcal L^{\rm cat}$ — the Ch 4 estimator. (b) $w_G\to0$ and $p_i \to
> \mathcal L^{\rm comp} = B^{\rm num}/\beta_{\bar G}$ — a pure population-model inference
> with no catalogue information, which should be *wide but centred*. Checking that second
> limit is not academic: the project's harness measured the pure-completion estimator to be
> **biased HIGH by +0.7…+5.4% in h** at deep incompleteness with 68% coverage collapsing
> (ledger #57). A limit that is supposed to be trivially safe was not.

**Q5.4** Split the events into in-catalogue (76) and dark (1512) and each class's summed
log-posterior peaks in a different place — **0.86 and 0.64**, with the combined answer at
0.73. Give two mutually exclusive readings of that, and say what would distinguish them.
> **Answer.** Reading A: both branches are fine and the classes genuinely carry different
> statistical power, so their crossing is a legitimate combination. Reading B: at least one
> branch is tilted, and the crossing is a coincidence of two opposing errors — in which
> case the "measurement" is not centred on truth, it is balanced there. The distinguishing
> test is **leverage**: if reading A holds, small perturbations to the class composition
> should barely move the answer; if B holds, they should move it a lot. The project ran
> exactly this and found a ±1/√N Poisson reweight moves the MAP by up to **0.025** in the
> realistic venue versus **0.0000** idealized, with $dh^*/d\varepsilon$ leverage
> **1500–2400×** — strong evidence for reading B. That is claim C5, and Ch 11 is where it
> lands.

**Q5.5 (transfer)** Both branches depend on how precisely the GW pins the source. Up to
now the localization has been "an ellipsoid, somehow". Name two properties of that
ellipsoid you would need to know before you could trust either branch.
> **Answer.** Its *size* (how many galaxies are in it, hence whether the catalogue sum
> is dominated by one host or ten thousand impostors) and its *orientation/correlation*
> (whether the sky and distance errors are independent — they are not, because the LISA
> antenna pattern couples them, and treating them as independent mis-weights every
> candidate). Ch 6 opens the box.

**Trap 5.A — "The completion term is a fudge factor for incompleteness."**
> It is a full likelihood branch derived from the same population model and the same
> selection function as the catalogue branch, and it is where 84%+ of the 2D channel
> difference lives. Calling it a fudge is how you avoid noticing that its prefactor is the
> largest tilt in the inference (C10).

**Trap 5.B — "w_G is roughly the catalogue's completeness fraction."**
> It is a *detection-weighted, rate-weighted, h-dependent* posterior probability, and the
> project measured it at **2.3–2.5× the realized in-catalogue rate** (0.1215 vs 164/3135 =
> 0.0523, binomial z = −11.86). Treating it as "completeness" is exactly the confusion
> that hid claim C9 for months.

---

### Ch 6 — Opening the Black Box

**Q6.1** The Cramér–Rao bound is a bound. Under what conditions is using it *as* the
posterior width defensible, and what would you expect to break first as those conditions
fail?
> **Answer.** It is the leading-order Gaussian approximation to the likelihood around its
> peak, valid at high SNR where the log-likelihood is locally quadratic. This pipeline's
> SNR ≥ 20 threshold is chosen with that in mind, and the loudest events (SNR up to 1425)
> are extremely safe. What breaks first as SNR falls is not the width but the *shape*:
> non-Gaussian tails and multimodality in sky position, which matter enormously here
> because the candidate-host sum is an integral over exactly those tails. A 5% error in σ
> is harmless; a missing secondary sky mode is not.

**Q6.2** The distance is parameterized fractionally, $u = d_L/\hat d_L$. Why is that the
natural coordinate, and what would go wrong with absolute $d_L$?
> **Answer.** Because the GW amplitude determines $d_L$ multiplicatively — the fractional
> error is roughly SNR-set and roughly d_L-independent — so σ in absolute distance scales
> with distance itself. In fractional coordinates the covariance is approximately
> stationary across the population, which makes the stored CRB reusable and the Gaussian
> approximation better behaved. It also means the mean of the fractional coordinate is
> exactly 1 by construction, which is why the code's mean vector is
> $(\hat\phi,\hat\theta,1)$.

**Q6.3** A missing $dt^2$ made every SNR 10× too small. Without looking it up: what does a
factor 10 in SNR do to (a) the Fisher matrix, (b) the CRB σ, (c) the detectable population
depth?
> **Answer.** (a) The Fisher is quadratic in the signal, so 100× too small; (b) the CRB σ
> goes as the inverse square root of the Fisher, so 10× too large; (c) since SNR ∝ 1/d_L
> at fixed source, a 10× SNR deficit shrinks the horizon ~10×, and the project measured
> the mock population capped at $z \le 0.11$ instead of $\lesssim 1.5$ — a *different
> universe*, not a rescaled one. The lesson the book draws: a constant factor in an inner
> product is never "just a normalization", because normalizations that multiply a
> *threshold* change which data exists.

**Q6.4** Why does a covariance stamped in the wrong sky frame bias H₀ rather than merely
add noise?
> **Answer.** Because the candidate-host sum weights each galaxy by the Gaussian evaluated
> at *that galaxy's* coordinates. A frame error systematically mis-places the ellipsoid
> relative to the real large-scale structure, so the sum picks up the wrong galaxies — and
> since galaxies at different sky positions are at different redshifts, the wrong galaxies
> prefer a wrong h, coherently across events. The project measured host recovery 31 → 38
> of 60 and the MAP moving 0.860 → 0.730 from this alone (ledger #12).

**Q6.5 (transfer)** You now have a well-characterized ellipsoid. For EMRI-889 the
distance shell is thin to 8×10⁻⁵. Yet the book is about to claim the redshift is the
dominant error. Estimate the redshift precision implied by a catalogue photo-z at
σ_z/z ≈ 49% and compare.
> **Answer.** The distance contributes ~0.008% to the H₀ error; a 49% fractional redshift
> error contributes ~49%, roughly **6000× larger**. The GW measurement is, for these
> events, effectively exact and the catalogue is effectively unmeasured. Everything from
> here on is about handling a wide, badly-known redshift correctly — and Ch 7 shows that
> the *naive* handling does not merely widen the answer, it moves it.

**Trap 6.A — "Sky localization and distance are separate measurements."**
> They are correlated through the antenna pattern, and the correlation is in the
> off-diagonal of the stored 3×3 covariance. Factorizing them mis-weights every candidate
> galaxy, and the error is largest exactly for the loud, nearby, information-carrying
> events.

**Trap 6.B — "Numerical factors are bookkeeping."**
> $dt^2$. Ten times the SNR. A tenth of the universe.

---

### Ch 7 — A Redshift Is Not a Number

**Q7.1** Predict, before the reveal: you replace each catalogue redshift by a Gaussian of
width σ_z centred on the catalogue value, and integrate. Does the H₀ posterior (a) widen
symmetrically, (b) shift low, (c) shift high?
> **Answer.** It **shifts low**, and quadratically: the project measured
> $\Delta h = -C\sigma_z^2$ with $C \approx 17$–$20$, giving −0.0016 / −0.0064 / −0.023 /
> −0.046 at σ_z = 0.005 / 0.015 / 0.035 / 0.050. The mechanism is Eddington's: more volume
> lies at higher z than lower z, so a symmetric error in *observed* z corresponds to an
> asymmetric posterior in *true* z, and using the bare Gaussian mis-states which side the
> galaxy is really on. Symmetric widening is what almost every reader predicts, and it is
> the trap this chapter is built around.

**Q7.2** Why is $p(z_g\mid z)$ "not a posterior", and what exactly does the volume prior
supply?
> **Answer.** $p(z_g\mid z)$ is a *likelihood* — a statement about the measurement given
> the truth. Read as a function of z at fixed $z_g$ it is numerically a Gaussian in z, and
> that numerical coincidence is what makes the mistake so easy. To weight candidate
> redshifts you need $p(z \mid z_g) \propto p(z_g\mid z)\,p(z)$, and the prior $p(z)$ is
> the *rate* of EMRIs per unit true redshift, $\propto (dV_c/dz)/(1+z)$: comoving volume
> per unit redshift, with a time-dilation factor converting source-frame rate to detector
> frame. That is the `volume_deconv` kernel, and G2b proves it is the unique weight
> consistent with this project's own rate model.

**Q7.3** Why must the same volume prior appear in the per-galaxy **denominator** as well
as the numerator, and what is the name of the invariant that enforces it?
> **Answer.** Because the numerator and denominator are integrals over the same true-z
> variable with the same measure; if the prior appeared in only one, the ratio would carry
> a spurious $dV_c$ and become h-dependent for a purely bookkeeping reason. G2b calls this
> "$dV_c$ counted once" and verifies it explicitly, including the exact h-invariance of
> the per-galaxy normalization $Z_g \propto h^{-3}$ to 1e−15. The project protects it with
> a binding regression gate — which is why any future fix to this kernel must *explicitly
> supersede* G2b rather than quietly contradict it.

**Q7.4** The deconvolution fixes an over-correction low. Now push it: what happens if you
deconvolve a *very* wide photo-z against a monotonically rising volume prior with **no**
selection turnover?
> **Answer.** You over-correct the other way. With no $p_{\rm det}$ and no catalogue
> selection in the weight, nothing penalizes pushing the host to higher z, so the
> deconvolved estimate inflates. The measured law on this code is
> $h_{\rm eff}/h_{\rm true} = \frac12[1+\sqrt{1+12(\sigma_z/z)^2}] \to 3(\sigma_z/z)^2$,
> with a rail threshold at **σ_z/z > 0.256**; the campaign's in-catalogue hosts sit at
> 0.25–0.49, predicting +16% to +49% inflation — and 58% of them do in fact rail at the
> top of the prior. This is claim **C7**, adjudicated as a MEASURED finding on the code's
> own numerator, and it is the book's cleanest example of a fix and its failure mode being
> the same object.

**Q7.5** Order of magnitude: spectroscopic redshifts are 0.56% of GLADE+. If you kept only
spec-z hosts, would the photo-z problem go away?
> **Answer.** No, and the project tested it. Spec-z hosts carry at most 8.7% (median ~0%)
> of the rate-weighted in-catalogue likelihood, and an inference-side spec-z-only cut
> **still railed at 0.870** (ledger #42). The problem is not that some redshifts are bad;
> it is that the estimator's treatment of *width* is wrong, and removing the wide ones
> removes almost all the data without removing the defect.

**Q7.6 (transfer)** The redshift is wide and its treatment is delicate. The GW measures the
*redshifted* mass $M_z = M(1+z)$ to about 10⁻⁴. Propose how that could be used, and guess
whether it will help.
> **Answer.** If a catalogue galaxy reports a black-hole mass $M_g$, then $M_z/M_g - 1$
> estimates z independently of the distance — a second handle, and an extremely sharp one
> on the GW side. It should sharpen host identification enormously, and mechanically it
> does: 97–99% of impostors are rejected. Whether the *posterior* improves is Ch 8, and
> the answer is the most instructive "no" in the book.

**Trap 7.A — "Bigger σ_z just means a wider posterior."**
> Q7.1. It means a *shifted* posterior, quadratically in σ_z, and the shift is ~17–20×
> σ_z² in h.

**Trap 7.B — "Deconvolution is the safe, principled choice, so more of it is better."**
> The same operation that removes a −0.023 bias at σ_z = 0.035 creates a +16–49%
> inflation at σ_z/z ≈ 0.25–0.49. Principled operations have domains of validity.

---

### Ch 8 — A Second Handle: the Mass Channel

**Q8.1** The GW measures $M_z$ to ~10⁻⁴; the catalogue's black-hole mass proxy has
σ_lnM ≈ 1.28 (a factor ≈3.6). Predict the *shape* of the resulting mass window's rejection:
symmetric, one-sided low, or one-sided high?
> **Answer.** Strongly **one-sided low**. Because $\sigma_{M_z}/M_z \approx 10^{-4}$ is
> negligible against the catalogue's factor-3.6 scatter, the window's width is set almost
> entirely by the catalogue side, and the physically-allowed condition $M_z = M(1+z) \ge M$
> makes the upper leg vacuous — a galaxy can always be "too light for its redshift" but
> never meaningfully too heavy. The project measured **193 low-side rejections against 1
> high-side**. Asymmetric rejection means asymmetric surviving support, which means a
> tilt.

**Q8.2** Rejecting impostors is what the 2D channel is *for*, and the 2D bias is +0.077.
Reconcile those two statements.
> **Answer.** The rejection is doing its job; the problem is what happens to the mixture
> afterwards. Deleting impostors removes a *down-pulling* catalogue leg without touching
> the up-tilting prefactor, so the mixture's balance shifts. The project's exact
> accounting: the channel difference for dark events is
> **+15.83 = 0 (completion, cancels identically) + 19.10 (loss of the 1D catalogue
> down-tilt) − 3.27 (residual 2D tilt)**, and the dark class's mean catalogue mixture
> weight falls **0.0354 → 0.0061**, a factor 5.8. The 2D channel did not add a wrong term;
> it *removed a right one's influence*.

**Q8.3** Why is "the 487 events whose 2D catalogue likelihood is exactly zero" **not** the
mechanism, despite being the most striking number in the dataset?
> **Answer.** Because their 1D legs were already negligible, so deleting them costs almost
> nothing: those 487 events carry **+0.24 nats = 1.5%** of the +15.83, and the 491
> both-dead events carry exactly 0.00. **98.5% is carried by the 534 survivors**, which
> were *de-weighted*, not deleted. This is the book's best example of a striking statistic
> being the wrong statistic — and it was the author-agent's own mechanism, refuted by
> exact algebra verified to 6.2e−13 on all 65,108 cells.

**Q8.4** You rescale your masses from solar masses to 10⁵ M☉. Which of the 1D and 2D MAPs
moves, and why is the answer a crisis rather than a curiosity?
> **Answer.** The 1D MAP is **bitwise invariant**; the 2D MAP walks — **0.81329 → 0.78107
> → 0.74440 → rails at 0.600** for a mass-coordinate rescale of C = 1 / 0.3 / 0.1 / ≤0.01,
> at +0.031 per e-fold. It is a crisis because a published number that moves with an
> arbitrary unit choice is not a measurement of anything. The cause, located by
> adjudication, is a **measure mismatch between the two numerator legs**: the 2D catalogue
> leg carries exactly one mass density while the completion leg carries none, so the
> arbitrary constant fails to cancel. Note the subtlety — a *consistent* physical unit
> change of all inputs **is** exactly invariant; what is broken is that the code silently
> hard-wires the measure to each event's own measured mass $M_{z,\det,i}$, which spans a
> factor 12 across the population.

**Q8.5** A mass-kernel improvement was confirmed in isolation (+0.016…+0.02, correct sign)
and then exonerated in the full pipeline (Δ2D = +0.0029, wrong sign; Δ1D = 0.0000
*exactly*). What single structural fact explains both results?
> **Answer.** The same prior enters the per-galaxy selection denominator $D_g$ as the
> numerator, so $N_g/D_g$ cancels the shift — "the selection denominator is not a
> spectator" (ledger #71 → #72). The exactly-zero 1D change is the tell: a change that
> touches only the mass channel *cannot* move 1D, so a nonzero 1D shift would have
> indicated a plumbing error rather than physics. The transferable lesson is that a toy
> that isolates the numerator is testing a different estimator from the one you ship.

**Q8.6 (transfer)** Every claim so far compares the estimator to "truth". Where did truth
come from, and name one way the *simulator* could make an estimator look biased when it is
not.
> **Answer.** From a forward model: a population drawn from a rate model, waveforms
> generated, detection applied, hosts drawn with a rate weight. The simulator can make a
> correct estimator look biased whenever the two disagree about the population — for
> instance if the simulator draws hosts *mass-aware* while the estimator's mixture weight
> is computed *mass-blind*. That is not hypothetical: it is claim C9, measured at a factor
> 2.3 with binomial z = −11.86, and it is Ch 9.

**Trap 8.A — "More information cannot hurt."**
> More information *in a correctly specified model* cannot hurt. In a model with a measure
> mismatch (C8) and a mis-calibrated mixture weight (C9), a sharper likelihood makes the
> mis-specification more decisive. The 2D channel is 10/10 runs at +4.04σ; the 1D channel
> is 0/10 exceeding 2σ.

**Trap 8.B — "The completion term pulls the answer up."**
> The **(1−w_G) prefactor** pulls up: N·Δln(1−w_G) = **+31.55** nats over h = 0.73→0.81.
> $\mathcal L^{\rm comp}$ itself pulls **down** for dark events (**−22.72** nats; only
> 39.1% of dark events have a positive completion tilt). The project created claim C10
> specifically to retire this sentence; the book must not resurrect it.

---

### Ch 9 — Building a Universe to Break Your Estimator

**Q9.1** The pipeline uses $\Omega_m = 0.2726$, not the Planck value. Bug or design choice,
and how would you tell?
> **Answer.** Design choice, and you tell by asking what the number is *for*: it matches
> the Barausse (2012) M1 cosmology that the EMRI population itself is drawn from, so the
> mock universe is self-consistent. Using Planck's $\Omega_m$ with an M1 population would
> put the generator and its own rate model in disagreement — a worse error than a
> known offset. The Planck mismatch is tracked as a *quoted systematic* (+1.5–3%), not
> silently absorbed. The general rule the chapter teaches: a constant is a bug only
> relative to a stated purpose.

**Q9.2** Two normalization modes give $w_G(0.73) = 0.1215$ and $0.0497$ for the *same run*.
Is one of them wrong?
> **Answer.** No — they are **different estimands**, and both are internally consistent
> within their own mode. `absolute_marginal` and `generator_marginal` normalize the
> catalogue likelihood differently, so "the probability the host is catalogued" means a
> different conditional in each. The danger is not the ambiguity but the *comparison
> across modes*: the project spent real effort chasing a phantom "45% discrepancy" that
> turned out to be two incompatible estimands placed side by side, and one diagnostics
> column was declared corrupt when it was merely the other mode's value.

**Q9.3** The estimator's $w_G(0.73) = 0.1215$; the realized detected in-catalogue rate is
164/3135 = 0.0523. Why is this a defect *independent of whether it causes the bias*?
> **Answer.** Because it is an internal inconsistency: the same code, in the same run,
> models a quantity at 2.3–2.5× what the same code's own generator realizes, at binomial
> z = −11.86. Whatever it does to the MAP, the estimator's mixture weight is not the
> quantity the simulation produced, and no downstream number can be defended while that
> holds. The project localizes the whole gap to the catalogue's *relative* detection
> efficiency — $\beta_G$ weights by a pool-marginal, population-mass $p_{\rm det}$, while
> Malmquist-selected catalogue hosts carry heavier black holes — with two independent
> suppression measures agreeing to 0.2σ (0.39248 vs 0.3991 ± 0.0312).

**Q9.4** The idealized campaign uses each host's *true* z and M point-to-point. Name what
that guarantees and what it therefore cannot test.
> **Answer.** It guarantees recovery-on-truth by construction — with the redshift exactly
> right, each golden event pins h at its GW distance precision, which is why the measured
> σ_h = 0.00030 matches "z exact + GW d_L error only" to ~5%. It therefore cannot test
> anything about redshift kernels, catalogue scatter, mass proxies, or the completion
> branch's calibration. It is a *consistency test of the generator–estimator pair*, and
> the readout says so in its own caveat section. The book quotes that caveat rather than
> celebrating the number.

**Q9.5 (transfer)** Your estimator recovers truth at −0.24σ on two independent seeds. State
the strongest claim you are entitled to make, and the claim you are not.
> **Answer.** Entitled: "on this venue, with these idealizations, the generator and the
> estimator agree, and no gross error is present in the path that connects them."
> Not entitled: "the estimator is unbiased" or "the estimator is calibrated" — the first
> needs many realizations, the second needs coverage. And there is a sharper trap: the
> project measured that **133 golden events carry ~100% of the curvature**, so the closure
> test validated *host association*, not the selection machinery it was assumed to
> validate. Ch 10.

**Trap 9.A — "The simulator is the easy part."**
> The simulator contains the population model, the plunge-window convention, the injection
> sampling measure, and the host-draw weight — four places where it can disagree with the
> estimator. Three of the book's live claims are exactly such disagreements.

**Trap 9.B — "If the generator and estimator disagree, the estimator is wrong."**
> Sometimes the generator is. The point of stating both is that "bias" is a *relation*
> between two models, and naming which one you are holding fixed is part of the claim.

---

### Ch 10 — Is It Calibrated?

**Q10.1** An estimator returns MAP = truth on two independent seeds at −0.24σ and −0.36σ.
List everything that still does not follow.
> **Answer.** It does not follow that the estimator is unbiased (two draws cannot resolve
> a bias smaller than the scatter — and note both are *negative*, which the readout itself
> flags as worth re-testing); that its stated uncertainty is correct (that is coverage,
> not location); that it will behave on a different venue (every negative result in this
> project is venue-scoped by standing rule); or that the parts you *think* were tested were
> tested — 133 golden events carrying ~100% of the curvature means the selection machinery
> was along for the ride.

**Q10.2** What does a P–P plot show that a bias measurement does not, and construct an
estimator that is unbiased and badly mis-calibrated.
> **Answer.** A P–P plot shows whether the *quantiles* of the posterior are honest: over
> many synthetic universes, the fraction of times truth falls below the posterior's q-th
> quantile should be q. An estimator that returns the correct mean but a σ half the true
> scatter is unbiased and catastrophically mis-calibrated — its 68% intervals contain truth
> ~38% of the time. The project measured exactly this shape in the pure-completion regime,
> with 68% coverage falling to 0.27, and in the noise-model study where coverage went
> 0.63 → 0.38 → 0.12 as N grew — the signature of a *real asymptotic bias*, since a purely
> noisy estimator's coverage does not degrade with N.

**Q10.3** The coverage harness measures the completion leg's deep-venue mis-calibration at
+0.0008…+0.0097 in h. The 2D bias is +0.077. What have you learned, and what have you
*not*?
> **Answer.** Learned: the completion leg's calibration is **6–16× too small to own the 2D
> bias** — a quantitative exoneration, monotone in completeness fraction across the full
> 0.008–0.85 range with no sign flip and control-consistent at zero. Not learned: anything
> about the 2D channel, because the harness is **1D-only / single-channel by construction**
> and has never covered the 2D residual. The book makes a point of this: the most useful
> thing a harness can tell you is often the *scope* of its own answer.

**Q10.4** Why is "coverage measured on synthetic universes" not circular, given that the
synthetic universes come from the same population model the estimator assumes?
> **Answer.** It is not circular for the failure modes it can see — kernel mis-specification,
> selection mis-scoping, quadrature, bookkeeping — because those break even when the
> population model is shared. It **is** blind to a mis-specified population model, since
> both sides would be wrong identically. That is why the project separately tests
> generator–estimator agreement (Ch 9) and why C9 could only be found by comparing the
> model's $w_G$ to the *realized* rate rather than by any coverage test.

**Q10.5 (transfer)** Your estimator passes the idealized closure and its completion leg is
exonerated quantitatively. Name the three things campaign #51 → #53 changed simultaneously,
and say what you can conclude about which one caused the +0.077.
> **Answer.** Catalogue scatter (unscattered → realized), host-z kernel (point δ →
> `volume_deconv`) and normalization mode (`generator_marginal` → `absolute_marginal`) —
> and you can conclude **nothing** about which one, because three variables moved at once
> and no run anywhere varies the estimator at fixed catalogue. That is claim C6, confirmed
> by a one-file check, and it is the honest opening of Ch 11.

**Trap 10.A — "It reproduces, so it's right."**
> Reproducibility is a property of the code, not of the physics. Ledger #49a: an estimator
> reproduced MAP = 0.86 perfectly across every injected truth from 0.63 to 0.77.

**Trap 10.B — "A negative result is permanent."**
> Standing scoping rule: negative conclusions are **venue-scoped**. Two of the project's
> exonerations were measured on the same 494-event shallow subsample and are explicitly
> *not* cross-venue confirmed — a shared venue idiosyncrasy would have fooled both.

---

### Ch 11 — The State of the Art, Honestly

**Q11.1** 58% of in-catalogue hosts peak at the top of the prior. Give the strongest
argument that this is an artifact of the prior's upper bound, then say how the project
killed it.
> **Answer.** The argument: any bounded grid piles up probability at its edge, so a
> concentration at 0.86 could simply be truncation of a broad, uninformative profile. It
> was killed three ways: the railed profiles are genuinely **concave** on the uniform
> stretch (86–96% all-negative second differences, |d²| ~10¹¹× roundoff); top-K parabola
> vertices give finite implied peaks at $h_{\rm eff} = 0.93$–$1.05$, stable over K = 3–9 in
> all 10 runs with the extrapolator validated in-band; and an independent reconstruction on
> a grid extended to h = 2.4 found **interior** peaks at median ≈1.12. It is a clipped real
> runaway.

**Q11.2** The same claim carries a binding fair-framing amendment: per event the rail is
"cosmetic" (0.30–0.47 σ_event) but the class-summed displacement is +3.4 to +6.1 σ_class.
Explain how both can be true, and why quoting either alone is a misrepresentation.
> **Answer.** Both are true because they are different summary statistics of the same data:
> individually each event's peak sits well inside its own (very wide) uncertainty, so no
> single event is anomalous; but the displacements are **coherently same-signed**, so they
> add linearly in the class sum while the class uncertainty grows only as √N. Quoting only
> the per-event number ("it's within 0.5σ") hides a >3σ systematic; quoting only "58% rail"
> invites the reader to picture 58% of events being individually broken, which they are
> not. The project made carrying both mandatory for any write-up, and the book obeys.

**Q11.3** C7 says the host-z kernel omits selection. The ratified derivation G2b *confirmed*
that exact weight, without $p_{\rm det}$, as uniquely consistent — and protects it with a
regression gate. How can both be right?
> **Answer.** They are answers to different questions. G2b establishes that
> $(dV_c/dz)/(1+z)$ is the unique weight consistent with this project's rate model and its
> selection integrals, and that it is exactly h-independent and reduces correctly as
> σ_z → 0. C7 measures what that same weight *does* at finite σ_z/z ≳ 0.256 when the
> numerator's window is z-proportional: it inflates. The honest statement is that G2b's
> premise and the finite-σ_z numerator kernel are in tension, so **any C7 fix must
> explicitly supersede G2b** rather than quietly contradict it — and must reckon with the
> fact that the measured historical failure mode of deconvolution at large σ_z/z was
> **over**-correction, the opposite sign to where the proposed fix pushes.

**Q11.4** Of C7, C8 and C9, which would you fix first if you could fix exactly one, and
what is the argument *against* your own choice?
> **Answer.** A defensible first choice is **C8**, because it is a well-posedness failure
> rather than a bias claim — a number that moves with a unit choice cannot be published at
> all, and the fix is identified and priced (give the completion leg its missing dark-host
> mass likelihood; the pure measure part moves 2D by −0.058, agreeing with the constant-C
> sweep at C ≈ 0.135). The argument against: the *full* fix also carries a +19.0-nat
> population tilt that lands at 0.84917 — the wrong direction — and independently
> reproduces a previously exonerated endpoint to 3e−5. So the measure part is robust and
> the population part is model-dependent, and shipping them together would import a
> Babak-M1-dependent term into a well-posedness repair. That decomposition is exactly why
> the project has not shipped it.

**Q11.5** Three measured inconsistencies (C7, C8, C9) all sit on the completion/prefactor/
kernel side, and none convicts the catalogue leg. Why is that a *direction* and not a
verdict?
> **Answer.** Because "we found problems on side A and none on side B" is evidence about
> where we looked, not only about where the problems are — and the catalogue leg's mass
> de-weighting, the mechanism that carries the 2D difference, is the 2D channel's
> *intended function*, which makes it hard to convict on principle. The decisive
> instrument is the pre-registered 2×2 cell B (unscattered catalogue × the #53 estimator),
> the only configuration that varies the estimator at fixed catalogue. The project wrote
> down its prediction **before** the result existed ("B ≈ C in both channels") precisely so
> that a contrary outcome would be unmistakably a surprise rather than something to
> rationalize.

**Q11.6 (transfer — the book's last question)** You now know this estimator's three live
defects, its confounded attribution, and its 98-item history of falsified fixes. What is
the *first* thing you would do, and why is "propose a fix" the wrong answer?
> **Answer.** Run the control. Attribution is confounded because three variables moved
> together, so any fix proposed now would be aimed at an unidentified target, and the
> ledger is 98 rows of what that produces — including fixes that were right in isolation
> and wrong in the pipeline, right in sign and 4× wrong in magnitude, and one that merely
> relocated the tilt to a different branch (+94…+455 nats, 12/12 fail). "Propose a fix" is
> wrong because in a near-flat likelihood surface with 1500–2400× leverage, *many*
> ±10-nat interventions move the MAP a lot, so a fix that moves the answer toward truth is
> almost no evidence at all. The discipline the book has been teaching — pre-register,
> isolate one variable, check whether it has been tried — is the answer.

**Trap 11.A — "Now that we know the defects, the number just needs correcting."**
> C9's counterfactual moves the 2D mean 0.8123 → 0.7433 *and* the 1D mean 0.7321 → 0.6430.
> The correction that fixes one channel breaks the other. There is no scalar correction.

**Trap 11.B — "The 1D channel is fine; only 2D is broken."**
> C5 says the opposite: the 1D headline is the crossing point of two railed, opposing
> runaways, and its centredness is contingent on the same mis-calibration — remove C9's
> defect and 1D goes to 0.643. The project explicitly does **not** claim the 1D channel is
> trustworthy.

---

## Part 4 — Interactivity pedagogy

**The rule every interactive obeys.** An interactive is an **experiment**, not an
illustration. Each specification below states what the reader *manipulates*, what they
*observe*, the **AHA** it is engineered to produce, and **what is lost if it were static**.
If the "lost if static" line is weak, the figure should be a static figure — the book
should have fewer, better interactives, not more.

Every interactive is driven by **real project data** wherever real data exists. The
diagnostics CSVs (1588 events × 41 h × 7 term columns per run, ×10 runs) are pre-computed;
the browser only needs to slice, sum in log-space, and re-render. Nothing in this book
requires running the pipeline client-side.

### 4.0 Book-wide instruments

**BW1 — The Bias Ledger Rail** *(chrome, present from Ch 3)*
- **Manipulate:** nothing directly — it responds to the reader's position and to any
  sandbox toggle they set. Clicking a rung expands the artifact and its date.
- **Observe:** the current estimator's measured bias in h, as a moving marker on a fixed
  scale from −0.18 to +0.08, with truth at 0.
- **AHA:** the estimator is an object under construction with a live scorecard, and the
  scorecard does not monotonically improve.
- **Lost if static:** the entire felt-sense of a *build*. A table of biases in an appendix
  conveys none of it.

**BW2 — The Symbol Passport** *(chrome, all chapters)*
- **Manipulate:** hover or tap any symbol; click to pin to a personal glossary.
- **Observe:** definition, units, code site, ratifying derivation, status badge.
- **AHA:** each symbol is a *decision someone made and defended*, not notation.
- **Lost if static:** a 40-symbol glossary at the back is not read. In-place, it is.

**BW3 — "Has this been tried?"** *(available inside every Break-It-Yourself sandbox)*
- **Manipulate:** the sandbox's own controls; plus a free-text search over the 98-row ledger.
- **Observe:** when the reader's configuration matches a historical hypothesis, the ledger
  volunteers the verdict, date, decisive test and residual.
- **AHA:** the reader's clever idea has a name, a date, and an obituary.
- **Lost if static:** the confrontation. A read-only ledger is a document; a ledger that
  interrupts you is a research supervisor.

### 4.1 Per chapter

**Ch 0**
- **I0.1 — The Arbitration Budget.** *Manipulate:* the σ and the unknown systematic of a
  hypothetical third method. *Observe:* whether it separates 67.4 from 73.0 at 3σ.
  *AHA:* precision without systematics control arbitrates nothing — the budget is
  dominated by the term you cannot measure. *Static loss:* the reader would take "we need
  1%" as the lesson instead of "we need an honest systematic".

**Ch 1**
- **I1.1 — Amplitude, Phase, Distance.** *Manipulate:* d_L, inclination, sky position of a
  synthetic EMRI. *Observe:* the strain time series and its measured amplitude; the reader
  tries to read d_L off the amplitude and cannot, because inclination mimics it.
  *AHA:* distance is recovered from the *joint* fit, not from amplitude.
  *Static loss:* the degeneracy is invisible in a single plotted waveform.
- **I1.2 — Meet EMRI-889.** *Manipulate:* nothing yet — a dossier card the reader opens.
  *Observe:* M = 7.25×10⁵ M☉, μ = 10 M☉, d_L = 88.9 Mpc, SNR 1425, σ_dL/dL = 8.0×10⁻⁵,
  and an empty slot labelled **z: unknown**. *AHA:* the empty slot is the book.
  *Static loss:* none — this one is deliberately static, and the dossier's *growth* across
  chapters is the interaction.

**Ch 2**
- **I2.1 — The Event Stacker.** *Manipulate:* add events to the stack, in random order or
  sorted by SNR. *Observe:* the combined log-posterior; and a live "effective N" readout.
  *AHA:* random order shows the curve barely moving for hundreds of events and then
  lurching — because the information is in 76 of 1588, and 3 of those carry 46%.
  *Static loss:* the lurch. A final posterior shows a nice peak and hides that it is three
  galaxies.
- **I2.2 — Bias / Scatter / Coverage Trainer.** *Manipulate:* two sliders (true bias, true
  σ) and a "stated σ" slider. *Observe:* 200 simulated repeats, their mean, their scatter,
  and the realized coverage of the stated 68% interval. *AHA:* the three failure modes are
  independent and you can see an unbiased estimator fail badly. *Static loss:* the reader
  cannot build the counterexample themselves, and this is the vocabulary the whole book
  runs on.

**Ch 3**
- **I3.1 — The Sky-Ball Explorer** *(predict-then-reveal)*. *Manipulate:* an h slider;
  optionally the sky-area and distance-precision of the event. *Observe:* the localization
  shell projected on a real GLADE+ patch, with candidate galaxies lighting up and each
  galaxy's rate weight shown as its size. The reader is asked to **place a marker on the
  galaxy they think is the host before revealing it.** *AHA:* sweeping h sweeps the shell
  across a *different galaxy population* — the mechanism, made physical — and the reader's
  marker is usually wrong, which is Ch 5's premise earned rather than asserted.
  *Static loss:* everything. A single-h picture cannot show a mechanism that *is* the
  h-derivative.
- **I3.2 — Ratio of Sums vs Sum of Ratios.** *Manipulate:* a toggle between the two forms,
  on the real candidate set of one event. *Observe:* the two likelihood curves and their
  MAPs. *AHA:* an "obviously equivalent" rearrangement is worth 0.010 in h.
  *Static loss:* the reader would not believe the size of the effect without moving it.

**Ch 4**
- **I4.1 — Delete the Denominator** *(break-it-yourself, predict-then-reveal)*.
  *Manipulate:* a three-way switch — D(h) **off** / **local window** / **full volume** —
  applied to the real seed-61000 event set. The reader must first drag a "where will the
  MAP go?" marker. *Observe:* the combined posterior redrawing; the MAP snapping from the
  bottom rail to 0.73. *AHA:* selection is not a correction, it is what makes the
  likelihood a likelihood; and the failure mode is a *rail*, not a nudge — a shape most
  readers do not predict. *Static loss:* the prediction step, which is where the learning is.
- **I4.2 — The Horizon Breather.** *Manipulate:* h. *Observe:* the detection horizon
  breathing in and out over the population, with the detected fraction and D(h) plotted
  alongside. *AHA:* D(h) is not an abstract normalization — it is a *visible volume*.
  *Static loss:* "D(h) is an integral over the population" stays a sentence.

**Ch 5**
- **I5.1 — The Two Branches.** *Manipulate:* the completeness fraction, and a toggle to
  show each branch's contribution separately. *Observe:* $w_G(h)$ falling 0.1625 → 0.0947,
  and the two branches' log-likelihoods. *AHA:* $w_G$ has a *slope*, and the slope is a
  tilt on the posterior independent of either branch's shape.
  *Static loss:* the slope is the point and a single-h bar chart has no slope.
- **I5.2 — The Two Runaways (plant).** *Manipulate:* a class filter — in-catalogue (76) /
  dark (1512) / both. *Observe:* three summed log-posterior curves peaking at **0.86**,
  **0.64** and ~0.73. *AHA:* the headline number is a *crossing*, and the reader is told to
  remember this. *Static loss:* the reader must *do* the split themselves for the crossing
  to feel like their own discovery rather than a claim; and the same widget returns in
  Ch 11 with the leverage control unlocked, which only works if they built it once.

**Ch 6**
- **I6.1 — The Fisher Ellipse Forge.** *Manipulate:* SNR, sky position, mission duration,
  and a "assume sky ⟂ distance" toggle. *Observe:* the 3×3 covariance rendered as a sky
  ellipse and a distance profile, with the candidate-galaxy count inside it.
  *AHA:* switching off the correlation visibly changes *which galaxies are candidates* —
  the covariance is not a summary statistic, it is a selection rule.
  *Static loss:* the toggle. The correlation's consequence is only legible as a difference.
- **I6.2 — The dt² Switch.** *Manipulate:* one binary switch. *Observe:* every σ ×10, the
  SNR threshold slicing away the population, the horizon collapsing, the depth going
  z ≲ 1.5 → z ≤ 0.11 — **and the whole galaxy catalogue view emptying out.**
  *AHA:* a Riemann-sum factor is not bookkeeping; it is a different universe.
  *Static loss:* the *scale* of the change. Two static panels read as "a bit worse".
- **I6.3 — The 4π Marginal.** *Manipulate:* toggle "evaluate at peak sky density" vs
  "marginalize over 1/4π". *Observe:* $B^{\rm num}$ jumping ~5000× and the posterior
  railing at the grid edge. *AHA:* the difference between evaluating and integrating is
  not a factor of order one. *Static loss:* the reader needs the toggle to believe 5000.

**Ch 7**
- **I7.1 — The Eddington Machine** *(the chapter's centrepiece, predict-then-reveal)*.
  *Manipulate:* (a) σ_z; (b) kernel = **bare Gaussian** / **volume_deconv**; (c) a
  **"selection turnover"** toggle that adds a $p_{\rm det}$-like rollover to the weight.
  *Observe:* three stacked panels — the photo-z likelihood, the volume prior
  $(dV_c/dz)/(1+z)$, and their product with its mode marked; below, the resulting H₀
  posterior; below that, a board plotting measured Δh against σ_z with the project's four
  real points (−0.0016 / −0.0064 / −0.023 / −0.046) and the −Cσ_z² law.
  *AHA (three-stage, in one widget):* (1) the bare kernel shifts the answer *low*,
  quadratically — most readers predicted symmetric widening; (2) `volume_deconv` lands the
  measured points on the law and repairs it; (3) **turning the turnover off at large
  σ_z/z over-corrects the other way**, past the rail threshold σ_z/z = 0.256 — the fix and
  its failure mode are the same knob. *Static loss:* the three-stage arc is a single
  continuous gesture on one control; split into three figures it reads as three unrelated
  facts.
- **I7.2 — Spec-z Rescue Attempt.** *Manipulate:* a "keep only spec-z hosts" cut.
  *Observe:* the sample collapsing to 0.56% of the catalogue and the posterior **still
  railing** at 0.870. *AHA:* the defect is in the treatment of width, not in the wide data.
  *Static loss:* the reader wants to try this; letting them try it and fail is worth more
  than telling them.

**Ch 8**
- **I8.1 — The Impostor Sieve.** *Manipulate:* the catalogue mass-proxy scatter σ_lnM, and
  a 1D/2D channel switch. *Observe:* the candidate galaxies of one real event being culled
  (97–99% at the true σ_lnM ≈ 1.28), the rejection histogram showing it is **one-sided**
  (193 low vs 1 high), and the surviving legs' mixture weight falling 0.0354 → 0.0061.
  *AHA:* the sieve works *and* the weight collapses — two things happen and only one of
  them is the intended one. *Static loss:* the simultaneity, which is the mechanism.
- **I8.2 — The Reparametrization Walk** *(the book's most alarming widget)*.
  *Manipulate:* a single control labelled in *units*: "measure black-hole masses in
  M☉ / 10⁵ M☉ / 10⁶ M☉ / kg". *Observe:* the 2D MAP walking **0.81329 → 0.78107 → 0.74440
  → rails at 0.600**, with the 1D MAP displayed alongside as **bitwise unchanged** and a
  running "d MAP / d ln C = +0.031 per e-fold" readout. A second control offers "change
  the units *consistently everywhere*", under which the MAP does not move at all.
  *AHA:* the failure is not "units matter"; it is that the two numerator legs carry
  *different mass dimensions*, so an arbitrary constant fails to cancel — and the code has
  silently hard-wired the measure to each event's own measured mass.
  *Static loss:* a table of four MAPs is a curiosity. Turning a units dial and watching a
  published number move is an alarm.
- **I8.3 — EMRI-889's Two Faces.** *Manipulate:* a 1D/2D switch and an event selector
  (889 = golden in-cat, 606 = dark). *Observe:* for 889, a clean 1D peak at h ≈ 0.745–0.750
  becoming a **nearly flat** 2D curve topping out at 0.79–0.80; for 606, a flat 1D
  catalogue leg, an 80×-suppressed and *decreasing* 2D leg, and a monotonically rising
  completion contribution. *AHA:* the whole class-level story is legible in two events, and
  the reader has been following one of them since Ch 1. *Static loss:* the side-by-side
  toggle is what makes the two events *comparable* rather than two anecdotes.

**Ch 9**
- **I9.1 — The Universe Factory.** *Manipulate:* population parameters (mass band, rate
  normalization, Ω_m), the plunge-window convention (uniform-in-plunge-time vs a
  snapshot p₀ draw), and the mission duration. *Observe:* the detected population's (z, M,
  SNR) distributions, and a warning when the *estimator's* assumptions no longer match.
  *AHA:* the snapshot p₀ draw silently freezes every $M_z \gtrsim 10^{6.2}$ event outside
  its detectable plunge phase — an "input-validity domain" used as an astrophysical prior.
  *Static loss:* the mismatch warning, which only means something if the reader caused it.
- **I9.2 — The Consistency Bench** *(predict-then-reveal)*. *Manipulate:* two dials — what
  the **generator** believes about host masses (mass-aware draw) and what the **estimator**
  believes when it computes $w_G$ (mass-blind vs mass-aware). *Observe:* the model $w_G(h)$
  curve against the *realized* detected in-catalogue rate, with a live binomial z.
  The reader is asked to make them agree. *AHA:* the shipped configuration reads
  **0.1215 vs 0.0523, z = −11.86**, and the only dial that fixes it is the one nobody
  turned — mass-awareness in $\beta_G$ (mass-aware $w_G$ = 0.05149, z = +0.21).
  *Static loss:* "the estimator and generator disagree" is abstract until the reader has
  personally failed to reconcile them.

**Ch 10**
- **I10.1 — The P–P Slot Machine.** *Manipulate:* completeness fraction and σ_z; press
  "run 200 universes" (precomputed grid). *Observe:* the P–P curve bowing away from the
  diagonal, the 68% coverage number, and the bias — *separately*. *AHA:* the bias can be
  small while coverage collapses, and coverage that **degrades with N** is the signature of
  a real asymptotic bias rather than noise (0.63 → 0.38 → 0.12).
  *Static loss:* the reader must see coverage and bias move *independently* to internalize
  that they are different things.
- **I10.2 — What Did the Closure Test Actually Test?** *Manipulate:* remove the top-K
  golden events from the closure sample. *Observe:* the constraint evaporating — 133
  events carry ~100% of the curvature. *AHA:* a passing test can be passing for a reason
  you did not intend; this one validated host association, not the selection machinery.
  *Static loss:* the removal gesture *is* the argument.

**Ch 11**
- **I11.1 — The Two Runaways, Unlocked** *(payoff of I5.2)*. *Manipulate:* the class
  reweighting slider (±1/√N Poisson), a λ-scan on the class balance, and an
  idealized/realistic toggle. *Observe:* the combined MAP sliding by up to **0.025** in the
  realistic venue and **0.0000** in the idealized one; the leverage readout
  $dh^*/d\varepsilon$ at 1500–2400× idealized. *AHA:* the reader's own hands move the
  headline number, and the same gesture does nothing to the idealized estimator — which is
  what "the answer is a crossing, not a peak" actually means. *Static loss:* the whole
  point is agency; a printed sensitivity number is forgettable.
- **I11.2 — The Adjudication Board.** *Manipulate:* filter the claim set C1–C11 by status
  (FINDING / REFUTED / CONFOUNDED / OPEN), by provenance tag ([LOCAL] / [DOC] / [INFER]),
  and by which side of the estimator they sit on. *Observe:* the claims arranged with their
  refutation routes, what was executed, and what remains. *AHA:* three measured
  inconsistencies all sit on one side — a *direction*, and the board shows why that is not
  a verdict. *Static loss:* the status filter is what turns a long document into an
  argument the reader can interrogate.

**Defect Museum**
- **M1 — The Falsified Fix Gallery.** *Manipulate:* pick an exhibit; each carries its own
  live control. The flagship is `volume_trunc`: a **quadrature-order dial**. *Observe:* at
  n = 50, `fixed_quad` returns **0.0000** for an integral whose exact value is 0.24–0.65 —
  the GW peak falls between nodes and is aliased away — while the estimator's bias goes the
  wrong way by ~4× (1D 0.745 → 0.800). Turn the dial up and the integral reappears.
  *AHA:* quadrature is physics, and a pre-registered fix can fail for **two independent
  reasons at once**. *Static loss:* the aliasing is invisible unless you watch the integrand
  and the nodes together while the node count changes.
- **M2 — Archaeology.** *Manipulate:* a timeline scrubber over stored posteriors by commit
  date. *Observe:* an unbiased baseline on 2026-04-09 and the rail appearing ~2026-04-24
  when the Gray completeness machinery was switched on. *AHA:* "when did it break?" is a
  question with a *method*. *Static loss:* the scrub is the bisect.

### 4.2 Interaction-design principles the implementer must hold

1. **Never animate a transition the reader did not cause.** Every change of state is
   traceable to a control they moved. Auto-playing figures teach nothing.
2. **Predict-then-reveal is locked, not suggested.** The reveal control is disabled until a
   prediction is recorded. The prediction persists in the margin and is re-surfaced later.
3. **Real data or clearly-labelled toy — never in between.** Each widget carries a small
   provenance chip: `real: seed61000/real_r1` or `toy: analytic`. A toy that looks like data
   is a scientific-integrity failure in a book about scientific integrity.
4. **Every "break it" state is recoverable in one click,** and the recovery button is
   labelled with the *fix's name*, so the reader learns the vocabulary by using it.
5. **Log-space by default.** Every combined-posterior widget sums log-likelihoods; the
   linear posterior is a display option. Readers should see log-space as the native
   representation, because that is where the additive structure of evidence lives.
6. **Honour the grid seams.** Any widget doing numerical differentiation must respect the
   non-uniform h-grid (0.01 / 0.005 / 0.01) and must refuse to differentiate across a seam
   rather than silently producing garbage — and should *say so* when it refuses, which is
   itself a lesson.
7. **Accessibility is not optional:** every widget has a keyboard path, every colour
   encoding has a redundant non-colour encoding, and every widget has a "show me the
   numbers" table view.

---

## Part 5 — Chapter dashboard

| Ch | Title | Discovery statement (the failure that opens it) | Beats | Interactives | Q / traps |
|---|---|---|---|---|---|
| 0 | Two Numbers That Should Be One | Every rung of the local ladder is calibrated against another rung — there is no independent ruler | B1 B7 | I0.1 | 3 / 1 |
| 1 | A Ruler That Needs No Ladder | GR gives d_L absolutely — and says nothing at all about z | B1 B4 B7 | I1.1 I1.2 | 4 / 1 |
| 2 | Bayes, Once and For All | One distance and no redshift is not yet an inference; and N events do not sharpen like √N | B1 B2 B4 B6 B7 | I2.1 I2.2 | 5 / 2 |
| 3 | Which Galaxy? | The localization ball holds tens of thousands of galaxies and you cannot tell which is the host | B1 B2 B4 B6 B7 | I3.1 I3.2 | 5 / 2 |
| 4 | The Universe Only Shows You Its Loud Half | Run it and the MAP sits at 0.60 — the bottom of the prior | B1 B2 B3 B5 B6 B7 | I4.1 I4.2 | 5 / 2 |
| 5 | The Galaxy You Cannot See | For 1512 of 1588 events the host is not in the catalogue at all | B1 B3 B5 B6 B7 | I5.1 I5.2 | 5 / 2 |
| 6 | Opening the Black Box | "An ellipsoid, somehow" is no longer good enough — how big, and correlated with what? | B1 B3 B4 B6 B7 | I6.1 I6.2 I6.3 | 5 / 2 |
| 7 | A Redshift Is Not a Number | Catalogue redshifts are photometric at σ_z/z ≈ 49%, and plugging in the central value *moves* the answer | B1 B2 B3 B4 B5 B6 B7 | I7.1 I7.2 | 6 / 2 |
| 8 | A Second Handle: the Mass Channel | The mass channel should sharpen everything — and the bias goes to +0.077 | B1 B2 B3 B4 B5 B6 B7 | I8.1 I8.2 I8.3 | 6 / 2 |
| 9 | Building a Universe to Break Your Estimator | Every claim compared to "truth" — where did truth come from, and does it agree with the estimator? | B1 B3 B4 B6 B7 | I9.1 I9.2 | 5 / 2 |
| 10 | Is It Calibrated? | It recovers truth at −0.24σ. That is not the same as being right | B1 B2 B3 B5 B6 B7 | I10.1 I10.2 | 5 / 2 |
| 11 | The State of the Art, Honestly | Three measured inconsistencies, a confounded attribution, and the control still running | B1 B4 B6 B7 | I11.1 I11.2 | 6 / 2 |
| — | Defect Museum (annex + 5 interludes) | 98 hypotheses, and what each one cost | B5 | M1 M2 | — |

**Totals:** 12 chapters · 60 self-check questions · 21 misconception traps · 24 chapter
interactives + 3 book-wide instruments + 2 museum interactives.

---

## Part 6 — Writing rules the book must not violate

1. **No number without provenance.** Every quantitative statement carries an artifact
   reference, and the reader can see it. This is not scholarly ornament: the book's subject
   is an investigation in which several confidently-quoted numbers turned out to be two
   different estimands.
2. **Status badges are load-bearing.** `RATIFIED` and `CANDIDATE` are not the same word.
   The 2D pairing the book spends Ch 8 on is a **designated candidate** (RATIFY-M6:
   "necessary, not established sufficient"), and the book says so where it introduces it,
   not in a footnote.
3. **Never quote half of an amended claim.** C5's rail statistic without the fair-framing
   amendment, or C4's mechanism without its refutation, would be a misrepresentation of the
   project's own adjudicated position. Where an amendment is binding, the book carries both
   halves in the same paragraph.
4. **Do not retire a defect by explaining it.** C7, C8 and C9 are live. The book explains
   them and leaves them live.
5. **The exoneration list is binding on the book too.** If the book's prose implies a
   mechanism that the ledger has refuted — "the completion term pulls up", "the mass window
   deletes the impostors and that is the bias" — it is wrong, regardless of how well it
   reads. C10 and C4-amended exist to prevent exactly those two sentences.
6. **Venue-scope every negative result.** "X was exonerated" is always "X was exonerated on
   venue V" — the project's own standing scoping rule, and two of its exonerations share a
   single 494-event subsample.
7. **The reader must always be able to tell whose voice is speaking:** the book's
   pedagogical narrator, the project's ratified derivation, or the adjudicator. Three
   distinct typographic treatments, used consistently.
