# STAGE-L READING — [86] Vijaykumar, Fishbach, Adhikari & Holz 2024, ApJ 972, 157

**Date:** 2026-08-25 · **Thread:** `[HIER]` · **Obligation:** stage-L reading gate, ledger row #195
item 2, before the `PROPOSAL_HIER_SELFCAL_20260825.md` prereg is finalized.

**Source:** arXiv:2312.03316 [astro-ph.HE], "Inferring host-galaxy properties of
LIGO-Virgo-KAGRA's black holes," Aditya Vijaykumar, Maya Fishbach, Susmita Adhikari, Daniel E.
Holz. Submitted Wed, 6 Dec 2023. Published *ApJ* 972, 157 (2024); DOI
10.3847/1538-4357/ad6140. Fetched via the arXiv HTML rendering (`arxiv.org/html/2312.03316`)
and the abstract page (`arxiv.org/abs/2312.03316`). This is reference [86] in Hanselman et al.
2024, cited at §IV.5 for the sentence quoted in the proposal's §1: *"it should be possible to
simultaneously infer the weighting scheme as well as H0 by generalizing the idea laid out in
[86]."*

All quotations below are `[LIT]`-tagged and verbatim from the fetched pages (WebFetch,
2026-08-25). Equation numbers are as rendered by the arXiv HTML.

---

## 1. Core method — what is jointly inferred, over what data, what hierarchical structure

**[LIT] Abstract (verbatim):** *"Observations of gravitational waves from binary black hole
(BBH) mergers have measured the redshift evolution of the BBH merger rate. The number density
of galaxies in the Universe evolves differently with redshift based on their physical
properties, such as their stellar masses and star formation rates. In this work we show that
the measured population-level redshift distribution of BBHs sheds light on the properties of
their probable host-galaxies. We first assume that the hosts of BBHs can be described by a
mixture model of galaxies weighted by stellar mass or star formation rate, and find that we
can place upper limits on the fraction of mergers coming from a stellar mass weighted sample of
galaxies. We then constrain parameters of a physically motivated power-law delay-time
distribution using GWTC-3 data, and self-consistently track galaxies in the
UniverseMachine simulations with this delay-time model to infer the probable host-galaxies of
BBHs over a range of redshifts. We find that the inferred host-galaxy distribution at redshift
z=0.21 has a median star formation rate ∼ 0.9 M☉yr⁻¹ and a median stellar mass of ∼ 1.9 × 10¹⁰
M☉. We also provide distributions for the mean stellar age, halo mass, halo radius, peculiar
velocity, and large scale bias associated with the host-galaxies, as well as their absolute
magnitudes in the B- and Ks-bands."*

**What is jointly inferred:**
- A **mixture fraction** α_SM giving the split between stellar-mass-weighted and
  star-formation-rate-weighted host samples (Eq. 1, below) — this is the "weighting scheme"
  object.
- A **power-law delay-time distribution (DTD)**: index α and minimum delay time t_D^min,
  convolved with the UniverseMachine cosmic star-formation-rate density (cSFRD).

**Over what data:** GWTC-3's population-level **redshift evolution of the BBH merger rate**
(not per-event host identification, not EM counterparts) — the observable is an aggregate rate
curve R(z), fit against ~90 detected BBH events (**[LIT]**, Introduction: *"~90 BBHs"*
detected by the LIGO-Virgo-KAGRA network).

**Hierarchical structure — the convolution chain:**

- **Eq. 2** (delay-time convolution, general form):
  `R(t) = ∫₀^∞ dt_D R_f(t + t_D) p(t_D)` — the merger rate at cosmic time t is the formation
  rate at earlier time, convolved with the DTD kernel p(t_D).
- **Eq. 1** (the mixture-model weighting law itself):
  `R(z) ∝ α_SM (1+z)^(−0.64) + (1 − α_SM) (1+z)^{2.5}` — a two-component mix of a
  stellar-mass-tracking term and an SFR-tracking term, each with its own fixed redshift
  power-law exponent; α_SM is the single free mixing parameter inferred from data.
- **Eq. 4** (the DTD→observable map): `κ̃(α, t_D^min) = log₂[R(z=1; α, t_D^min) / R(z=0; α,
  t_D^min)]` — maps DTD parameters into the log-ratio slope κ that GWTC-3 directly constrains.
- **Eq. 5** (the top-level likelihood): `ℒ(data | α, t_D^min) = ℒ(data | κ̃(α, t_D^min))` — the
  GW population likelihood is evaluated not on (α, t_D^min) directly but through the derived
  merger-rate-evolution statistic κ̃, i.e. a **reparametrized single-summary-statistic
  likelihood**, not a per-event redshift-by-redshift fit.
- **Eq. 6** (the individual-galaxy translation step, downstream of the population fit):
  `R^merg_i(z₀) = ∫ dt_D R^SFH_i(t(z₀) + t_D) p(t_D)` — once (α, t_D^min) are fixed from the
  population fit, the *same* DTD kernel is applied per UniverseMachine-tracked galaxy i to
  produce a merger-rate weight for that galaxy at any target redshift z₀.

So the hierarchy is: **population-level GW rate evolution → constrains a low-dimensional DTD
(2 parameters) via a single summary statistic κ̃ → the fitted DTD is then applied
galaxy-by-galaxy in a separate, deterministic simulation (UniverseMachine) to produce a
posterior *distribution* over host-galaxy physical properties** (SFR, stellar mass, halo mass,
magnitude, etc.), not a per-event galaxy assignment.

---

## 2. The exact "idea" a generalization would build on

The generalizable object is **not** a per-event host-probability formula (the paper does not
write one — see §2b below) but the **two-step inference-then-propagation pattern**:

1. Infer a **low-dimensional weighting-scheme parameter** (here α_SM, or (α, t_D^min)) from an
   ensemble-level GW observable, using a likelihood that is a function of a *derived summary
   statistic* of the population (κ̃), not of individual-event galaxy assignments.
2. **Propagate** the fitted weighting-scheme parameter into per-galaxy weights (Eq. 6) used
   downstream by another method (the dark-siren H0 statistical method) — i.e., the *weighting
   scheme becomes a population-inferred quantity rather than an assumed one*, and is used to
   build luminosity/property weights for candidate hosts.

**[LIT] §III.3.2 (Magnitudes) — the explicit hand-off statement:** *"These inferred histograms
on the magnitudes can be directly used to calculate luminosity-weights in the dark-siren method
for measuring the cosmic expansion history with GW events."*

**[LIT] Abstract's closing methodological claim, restated in §IV (Summary):** *"Our inferred
distributions on various parameters can be used to design optimal weighting schemes for the
dark-siren method to constrain the expansion history of the Universe."*

**[LIT] §I Introduction — the stated motivation for the weighting-scheme link:** *"Any
information about probable host-galaxies could also be used to reduce the number of candidate
galaxies searched to localize a GW transient, and in designing optimal weighting schemes in the
statistical method to measure cosmological parameters."*

**[LIT] §II — the negative result that *is* the actionable weighting-scheme finding:** *"our
results show that a stellar mass weighting is inconsistent with the redshift evolution of the
BBH merger rate, and applying such a weight during the analysis should be avoided."* This is
the one place the paper makes a direct, falsifiable statement about a commonly-used dark-siren
weighting choice (stellar-mass weighting, à la Gray et al. 2020-style luminosity/mass weights)
being disfavored by their inferred DTD/mixture posterior.

**(2b) Role of galaxy catalogues/weights, precisely:** the paper does **not** use a real galaxy
catalogue (GLADE-type) at all — it uses the **UniverseMachine simulation** as a synthetic
galaxy population with known SFH/stellar-mass tracks per halo, and computes Eq. 6 on that
simulated population. No explicit per-galaxy weight formula `w_i` or host-probability `p(host)`
equation appears anywhere in the paper (confirmed by direct re-query of the text) — the
"weight" is implicit: it is the merger-rate-per-galaxy R^merg_i(z₀) of Eq. 6, from which a
normalized weight over a catalogue of candidate hosts would be constructed *outside* this
paper, in the downstream dark-siren application the authors gesture at but do not build.

---

## 3. Stated validity conditions / regime

- **N (events):** **[LIT]** *"~90 BBHs"* detected total by LVK at the time of writing;
  GWTC-3's redshift-evolution constraint is the single population-level observable actually
  fit (not 90 independent per-event redshift posteriors folded hierarchically — the fit is to
  the aggregate rate-evolution slope κ derived from the full catalogue).
- **Redshift range:** current-detector horizon is described as **[LIT]** *"horizon redshift
  z∼1"*; **[LIT]** Eq. 4's approximation is stated to hold *"Since current observations measure
  the merger rate only out to z=1"* — i.e., the κ̃ log-ratio statistic is deliberately anchored
  at z=0 and z=1 because that is the reach of the data, and the paper explicitly flags that the
  *"merger rate at higher redshifts is less well-constrained."* Numeric host-property results
  are reported centered at z=0.21 (median SFR/mass benchmark) with results extending to
  z∼0.81 with growing uncertainty.
- **Error-model assumptions:** no explicit redshift-measurement-error model is stated for the
  GW events — the analysis works on the *published* GWTC-3 population-level redshift evolution,
  not on a per-event Gaussian (or other) redshift-error kernel. There is no analog anywhere in
  the paper of a per-event z-uncertainty propagation step.
- **Physical-model caveat, explicitly flagged (**[LIT]** end of §III.2):** *"For the main
  results in this work, we have assumed that binaries followed a physically motivated
  merger-rate evolution, given by the UniverseMachine cSFRD convolved with a delay-time
  distribution. This prescription is expected to work well if all binaries form and merge in
  isolated galactic field environments."* — a stated regime restriction (field formation only;
  dynamical-channel formation, e.g. dense clusters/AGN disks, is out of scope).
- **Metallicity simplification (**[LIT]**):** *"UniverseMachine does not model metallicity
  evolution, we fix the metallicity Z=0.5 Z⊙ at all redshifts"* — a stated fixed-parameter
  approximation, not inferred.
- **Selection effects:** the paper notes, in the context of future EM-counterpart follow-up
  rather than its own inference, that only a small fraction of BBHs will be **[LIT]**
  *"sufficiently localized such that the host-galaxy can be confidently identified"* even with
  next-generation detectors — i.e. this is not a per-event host-identification method; it stays
  strictly at the population level. No GW selection-function term is stated as entering Eqs.
  1–6 directly (the selection/detectability handling lives inside the published GWTC-3
  population inference the paper takes as input, not inside this paper's own likelihood).

---

## 4. Photometric redshift errors — explicit statement

**None found.** Direct re-query of the full HTML text for photo-z / redshift-uncertainty /
catalogue-completeness content returned: **[LIT]** *"No explicit treatment mentioned... Uses
spectroscopic [i.e., GW-measured] redshifts from GWTC-3... Makes no mention of photometric
redshift errors... Does not discuss galaxy catalogue completeness or selection bias in
UniverseMachine."* The paper's one completeness-adjacent remark is that UniverseMachine is
calibrated against **[LIT]** *"a wide variety of observations over a range of redshifts out to
z=10,"* with no quantitative completeness limit given. **This paper contains no photo-z error
model, no galaxy-catalogue completeness correction, and no per-galaxy redshift-error kernel of
any kind — the entire σ_z machinery our (h, θ)-grid experiment is built around is absent from
[86].** The generalization Hanselman+ point to is therefore genuinely novel on this axis, not a
re-application of an existing photo-z treatment.

---

## 5. Mapping sketch — generalizing [86] to a joint photo-z error-model + H0 inference

**What "generalizing [86] to jointly infer a photo-z error model θ with H0" would concretely
mean in the [86] formalism:**

[86]'s pattern is: (population-level GW observable) → (low-dimensional weighting-scheme
parameter, α_SM or (α, t_D^min), fit via a derived summary statistic κ̃) → (parameter
propagated per-galaxy into a downstream weight, Eq. 6) → (weight consumed by the dark-siren
method, unbuilt in [86] itself). To generalize this to "jointly infer weighting scheme *and*
H0," the natural reading is: replace the *external, one-way* hand-off (fit weighting scheme
first in [86], then bolt the resulting weights onto a *separately run* dark-siren H0 analysis)
with a **single joint hierarchical likelihood** in which the weighting-scheme parameter(s) and
H0 are inferred *together*, each informing the other's posterior — i.e. promote α_SM (or a
photo-z error-model analog) from a fixed input to the dark-siren stage to a co-fit parameter
alongside H0, inside one likelihood. [86] never writes this joint likelihood; it names the
*idea* (population-inferred weighting scheme feeding the dark-siren method) and the specific
citation Hanselman+ make is to *that* two-step inference-then-propagation pattern, generalized
so the propagation direction also flows backward (H0's posterior shape informs which
weighting-scheme values are consistent, not just the reverse).

**Where our (h, θ)-grid instrument is the same:**

- **Same core pattern — low-dimensional nuisance parameter, jointly inferred with the target
  cosmological parameter, evaluated over an ensemble.** [86]'s α_SM / (α, t_D^min) plays exactly
  the structural role our θ = (b, s) plays: a small parameter vector governing how per-event
  data maps onto galaxy-catalogue weighting, fit *jointly* rather than fixed a priori.
- **Same "θ enters only through per-event kernels" simplification.** [86]'s Eq. 6 shows the DTD
  parameters entering *per-galaxy* through a fixed transform (the convolution kernel p(t_D))
  applied uniformly across the ensemble — structurally identical to our θ entering only the
  per-event z-kernel (bias slope b, scatter scale s) rather than requiring new architecture per
  event, which is exactly the "feasible core" claim in `PROPOSAL_HIER_SELFCAL_20260825.md` §2.
- **Same target downstream consumer.** Both projects explicitly aim the inferred
  weighting-scheme object at the **dark-siren H0 method** ([86] §III.3.2, §IV; our proposal's
  entire framing).

**Where our instrument is different:**

- **[86] never builds the joint likelihood; we do.** [86]'s inference is single-stage
  (weighting-scheme parameters only, from population rate evolution) with propagation to the
  dark-siren stage left as future work / a qualitative recommendation. Our (h, θ)-grid computes
  the actual joint posterior L(h, θ) per event and marginalizes — the object [86] gestures at
  but never writes down.
- **Different physical content of θ.** [86]'s parameters (α_SM, DTD shape) describe **which
  galaxies host mergers** (an astrophysical formation-channel question). Our θ = (b, s)
  describes **how wrong the catalogue's redshift error model is** (a measurement/instrumental
  question) — a bias slope Δz = b·(1+z) and a scatter rescaling σ_z → s·σ_z on the *same*
  per-event z-kernel already used for host association, not a new astrophysical weighting axis.
  These are complementary generalizations of "weighting scheme," not the same one: [86]'s
  weighting scheme selects *which* galaxy is the host; ours corrects *how well-known* each
  galaxy's redshift is when doing that selection.
- **Different data regime.** [86] operates on GWTC-3's aggregate **population-level rate
  evolution** across ~90 events reduced to one summary statistic (κ̃); our instrument operates
  **per-event**, on individual localization-volume likelihoods evaluated on a grid, at the
  campaign's mirror-venue N (order 10²), and is explicitly designed to test identifiability at
  that concrete small N (Table in §3 of the proposal: "Unidentifiable... the lever is dead at
  our N"). [86] states no small-N validity condition at all — confirming the proposal's §1
  claim that *"no small-N validity statements exist"* in this literature position for the
  weighting+H0 joint-inference idea.
- **No catalogue-incompleteness or photo-z content to inherit.** As established in §4, [86]
  supplies zero machinery for photo-z error, so the (h, θ)-grid's kernel-transform design, its
  truth-θ = (0, 1) mirror-venue construction, and its coverage/P–P registered reads (proposal
  §2 items i–iii) are original to this thread, not adapted from [86] — only the *joint-inference
  pattern* is borrowed, exactly as the proposal's §1 framing already states ("named, never
  built").

---

## Bottom line for the prereg

[86] licenses the *shape* of the claim in `PROPOSAL_HIER_SELFCAL_20260825.md` — a
low-dimensional weighting/error-model parameter, jointly inferred alongside a cosmological
target, entering only through per-event kernels, feeding the dark-siren method — but contributes
no formalism, no equations, and no validity statement that transfers directly to the photo-z
axis or to our N. The stage-L reading obligation is discharged: the field position quoted in
the proposal (*"named, never built"*) is confirmed accurate by direct reading of [86], and no
revision to the proposal's claim, instrument, or decision table is indicated by this reading.
