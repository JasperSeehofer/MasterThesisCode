# dv-jr1-transform — joint_r1's T2.2b-equivalent transform under the log-normal realized-forward mass law

Date: 2026-09-02. Node: dv-jr1-transform (Research Graph 1, Branch C, wave 1).
Authorization: ledger row #290 / decisions-table row 5 ([DO] "the joint_r1 T2.2b-equivalent
transform derivation + the r-jr1-massaware draft"). Analysis-only: **no repo code edited, no
commits, no cluster**. Every number carried from history cites its ledger row or artifact;
everything computed by this node is marked **DERIVED-HERE** with the script named.
Scripts + JSON outputs live in this directory: `derive_jr1_transform.py` (stage 1),
`diag_and_response.py` (diagnostics + response), `final_stage.py` (the numbers of record),
`jr1_transform_final.json`, `jr1_diag_response.json`, `jr1_transform_stage1.json`.

## 0. What is being derived, and why the iiib 1.039 does not transfer

T2.2b banked, for the iiib venue, the ARITH true-host transform
S_4D(z_true, M_true; h)/S̄_φ(z_true; h), median **1.039** over the 66 recovered true
in-catalogue hosts, h-stable (1.0394/1.0391/1.0388 at h = 0.725/0.730/0.735)
{row #282; `tree2_20260830/t2_2b_arm_b_run/T2_2B_RUN_RECORD.md`, transform table}.

The mass law the estimator faces is venue-dependent {row #270;
`tree2_20260830/PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §1.2/§1.5, as corrected by its
Revision note 3 / row #271}:

- **iiib**: the evaluation catalogue is the unscattered reduced catalogue → for a catalogued
  true host, M_cat = M_true to the byte — a **delta law**. The transform is a point read.
- **joint_r1**: the evaluation catalogue is `observed_catalogue_seed900001.csv`
  (`headreadout_20260827/joint_r1/run_metadata_21.json:cli_args.observed_catalogue`;
  confirmed identical in the joint_r1 and off_joint_r1 metadata read by this node) — a
  **log-normal realized-forward law**: ln M_obs = ln M_g + σ_lnM·N(0,1),
  σ_lnM = BH_MASS_ERROR/BH_MASS (`observed_realization.py:349-356`, cited via row #270 §1.2;
  exact-width writer confirmed for seed 900001, row #271).

So on joint_r1 the factor the flipped leg applies at a true host is evaluated at the
**scattered** catalogue mass, not the true mass; the transform is a functional of the
realized draw and does not equal the iiib point value. A fresh derivation is required
(graph proposal §1.3; state candidate 10).

## 1. The likelihood structure as implemented (every structural claim pinned to file:line)

All lines re-read from the working tree (branch `fix/p32d-classg-venue-repair`, post-flip
commit 5e7fda16 in history, row #286) by this node on 2026-09-02.

1. **The flag and its auto resolution.** `catalogue_leg_1d_mass_aware` ∈ {auto, off, on},
   class default "auto" (`darksiren_emri/bayesian_inference/bayesian_statistics.py:3695,
   :3786, :4036`); "auto" engages "on" iff the production φ-stack resolves
   (numerator+global-selection "phi", θ-divisor "off") — resolution block `:4410-4470`;
   explicit "off" logs COUNTERFACTUAL (`:4433`), "on" logs [PHYSICS] ACTIVE (`:4462`).
   The joint_r1 production configuration IS the φ-stack
   (`run_metadata_21.json:cli_args`: normalization_mode=absolute_marginal,
   catalogue_global_selection=phi — read by this node), so post-flip joint_r1 runs "on".
2. **Site N1, the factor.** `catalogue_leg_1d_mass_aware_factor` (`bayesian_statistics.py:7059-7160`):
   under `sigma4d_mass_kernel="point"` the per-node factor is
   S_4D(d_L(z;h), **M_g**(1+z), φ=0, θ=0; h) — the raw catalogue BH_MASS, delta kernel in M;
   `M_g_error` is read **only** under the "kernel" branch (`:7129-7143` point path;
   docstring `:7107-7110`). Call sites: quadrature path `:7653-7663`, generator-point path
   `:7715-7735` (scalar twin); batch kernel per the registration's §2.2 (:8086-8089 pattern;
   flag validated at `:8423`).
3. **Production mass-form.** `sigma4d_mass_kernel` defaults to "point" (`:3597, :3738`) and
   the joint_r1 run of record used `sigma4d_mass_kernel = point`
   (`run_metadata_21.json:cli_args`, read by this node). **Consequence: on joint_r1 the
   engaged leg point-queries S_4D at the OBSERVED (scattered) mass.** The catalogue mass
   error does not enter the factor; it entered the DATA when the realization was written.
4. **The assembled identity.** Registered form
   (`tree2_20260830/PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md` §2.3): the assembled
   p_i under "on" differs from "off" by exactly one factor per candidate,
   S_4D(d_L(z;h), M_g(1+z))/S̄_φ(z;h), inside the candidate sum; the divisor/weight
   re-booking (Σ_φ, β_G) → (Σ_4D, α_G) is an exact identity on combined_no_bh.
   **Numerically re-verified by this node on the T2.2b dumps (DERIVED-HERE,
   `diag_and_response.py`)**: per-event N_on/L_on and N_off/L_off imply single global
   divisors Σ_4D = 3.75453e8 and Σ_φ = 9.80867e8, each uniform across all 982
   candidate-bearing events to relative spread ≤ 1e-13; Σ_φ matches the run record's
   9.809e8 at h=0.73 (`T2_2B_RUN_RECORD.md` GATE R). Pure columns B_num, D̃_φ are
   exact-zero different on vs off (max_rel 0.0) — reproducing row #282's invariance.

**Transfer pin (why this derivation is joint_r1's own):** every structural element above is
common to both venues; the ONLY venue difference at the factor is the mass argument's law
(delta vs realized log-normal) plus the induced candidate-set differences (§5, caveat C4).
A transform derived for the iiib configuration therefore does not transfer *numerically*
(the argument's law differs) even though the *functional form* is shared — exactly the
row #270 finding the graph proposal encodes.

## 2. The joint_r1 transform: definition and derivation

**Definition (the T2.2b-equivalent object).** For the joint_r1 true in-catalogue hosts t
with true redshift z_t, true mass M_t, per-row width σ_t = BH_MASS_ERROR/BH_MASS, and
realized draw ε_t ~ N(0,1) frozen in seed 900001:

    T_jr1(h) = median_t [ S_4D(d_L(z_t;h), M_t e^{σ_t ε_t} (1+z_t)) / S̄_φ(z_t;h) ]

This is a **realization-conditioned** statistic (one frozen draw per host), the exact
analogue of what a T2.2b-style dump on joint_r1 will read. Its law under ε is what this
node derives; the delta-law limit σ_t → 0 must recover the iiib-type point median.

**Method (DERIVED-HERE, `final_stage.py`).** S_4D is venue-independent (same injection
pool/detection model both venues — both run_metadata_21 stamp the same injection, git
d04d9dc9, per row #270 §1.2) and is a deterministic function of (z, lnM) at fixed h:
verified on the 1.19M-row T2.2b off dump — within (Δz=5e-4 × ΔlnM=5e-3) cells the ln S_4D
std is median 0.003 (p95 0.025) (`diag_and_response.py`, "determinism"). The dump therefore
samples the S_4D surface densely, and the transform integral can be evaluated on it with
zero new physics compute:

- Per z-slab of 0.002, build a raw-point interpolant of ln S_4D vs lnM (duplicate-averaged,
  flat ends). Fidelity: the w·N-weighted dark-candidate aggregate of the interpolant
  reproduces the dump's own S_4D aggregate to 2e-5 (`fidelity_wN_aggregate` 0.99997-0.99998
  at all three h); true-host point fidelity median 1.0000.
- Smear with 41-node Gauss-Hermite in ε for expectations; Monte Carlo (4000 realizations of
  {ε_t}, seed 20260902) for the realized-median law. σ_t from the true-host rows' own
  M_err_g/M_g (median 1.276 — the iiib parent columns joint_r1's realization scatters with).

**Result — the derived transform (DERIVED-HERE):**

| h | delta-law median (check) | T_jr1 realized-median (MC q50) | 95% MC band | E-form median | K = E[S]/S median |
|---|---|---|---|---|---|
| 0.725 | 1.0394 | **1.0316** | [1.0215, 1.0364] | 0.951 | 0.9061 |
| 0.730 | 1.0391 | **1.0314** | [1.0211, 1.0363] | 0.951 | 0.9066 |
| 0.735 | 1.0388 | **1.0312** | [1.0211, 1.0362] | 0.951 | 0.9070 |

**Reading:** the joint_r1 T2.2b-equivalent transform is **≈ 1.031**, slightly BELOW the
iiib 1.039, with a realized-median predictive band **[1.021, 1.036]** (95%). The
mechanism: true hosts sit on the S_4D plateau (that is what made them detected events), so
most log-normal draws move them along a flat profile — the *median* barely moves — while
left-tail draws (host scattered light) fall off the plateau edge and drag the *mean*:
per-host realized ratio pooled quantiles (5/25/50/75/95%) = 0.440 / 0.959 / 1.031 / 1.047 /
1.081. The expectation-form median (0.951) and the realized-median (1.031) split is the
same mean-true/median-false structure T2.2b found in the dark class (row #282,
median-q REFUTED-IN-DETAIL) — here derived, not measured.

## 3. Dimensional analysis and limiting cases

- **Dimensions.** S_4D and S̄_φ are both dimensionless survival probabilities
  (`bayesian_statistics.py:2307`: "S_4D is dimensionless"; S̄_φ = ∫φ S_4D dlog10M over a
  normalized φ, `:2068-2069`). T_jr1 is a ratio of dimensionless quantities:
  **dimensionless**, as the iiib 1.039 is. The factor's arguments carry d_L [Gpc] and
  M_z [M_sun] into the same interpolator Σ_4D queries (`:7129-7143`) — no unit conversion
  is introduced by this derivation.
- **Limiting case (σ → 0, the delta-law/iiib-type result).** Registered requirement: the
  log-normal width → 0 must recover the point transform. Checked two ways (DERIVED-HERE):
  (i) analytically, the GH smear at σ = 0 is the identity, so T_jr1 → median_t S_4D/S̄_φ =
  the delta-law median by construction; (ii) numerically, smearing at σ = 1e-6 on the
  stage-1 surface returns 1.0474 against the same surface's own point median 1.0474
  (`jr1_transform_stage1.json`, D_sigma0_limit vs the surface σ→0 read) and the
  final-stage same-surface ratio K(σ=0) = 1 exactly; the dump point median is 1.0391.
  PASS.
- **h → h′ scaling / g-invariance of the assembly.** The transform must be invariant under
  the (Σ_φ, β_G) → (Σ_4D, α_G) re-booking (it is defined on the factor, which the identity
  isolates): verified numerically in §1 item 4 (global divisors uniform to 1e-13; pure
  columns untouched at 0.0). This is the g-invariance instrument the graph row names for
  this node.

## 4. h-stability (registered requirement, T2.2b parity)

T2.2b's 1.039 was h-stable across the 3 secant nodes (1.0394→1.0388, spread 6e-4; row #282).
The derived joint_r1 transform at the same nodes: **1.0316 / 1.0314 / 1.0312 — spread
4e-4** (realized-median q50), with the 95% MC band edges moving by ≤ 5e-4 across nodes and
the dark-class smearing ratio R_K moving 1.4045→1.4006 (spread 4e-3). **h-stable at the
same order as the iiib transform.** PASS.

## 5. The dark-class response ingredient and the proposed-band engine (feeds r-jr1-massaware)

The registered readout needs a posterior-level band, which requires the dark-class response,
not only the true-host transform. DERIVED-HERE (`final_stage.py`, `diag_and_response.py`):

- **Smearing RESCUES impostors.** On the w·N-weighted dark-candidate population, the
  log-normal smear raises the survival aggregate by **R_K = 1.40** (1.4045/1.4025/1.4006 at
  the 3 nodes): Jensen convexity on the S_4D cliff — a deeply suppressed impostor's
  scattered mass sometimes lands in the detection band. Survival-weighted ρ moves
  0.1035 → 0.1452. Direction: joint_r1's mass-aware annihilation of the impostor field is
  **weaker** than iiib's.
- **joint_r1 off-arm response, computed exactly on the banked 41-node grid**
  (`headreadout_20260827/joint_r1/event_likelihoods.csv`, 1588×41): trapezoid/flat-prior
  moments (the BAND_REDERIVATION §2.2 convention) give off MAP 0.600, mean 0.6143, floor
  mass 0.2208, ℓ′(0.73) = −255.03; per-event secant scores (0.725/0.735) split by the
  T2.2b per-event host map: dark Σ s_imp = −272.31 (n=1512), in-catalogue −179.50 (n=76),
  Σ s_pure = +196.78, dark-only Σ s_pure = −59.87.
- **Predicted on-arm posterior.** ℓ_on(h) = ℓ_off(h) + Δ′·(h−0.73) [+ ½Δ″(h−0.73)²], with
  Δ′ = (ρ_jr1 − 1)·(−272.31) + 1.7 (in-catalogue term scaled from iiib's measured +1.55,
  row #282) and Δ″ scaled from iiib's single second difference −905.03 by Δ′/216.9.
  Central ρ_jr1 = 0.2604 (iiib measured, row #282) × R_K 1.40 = **0.365** under the
  proportional-transfer assumption (caveat C2); scanned over ρ ∈ [0.26, 0.50] × {lin, quad}:

  | ρ_jr1 | lin MAP/mean | quad MAP/mean |
  |---|---|---|
  | 0.26 | 0.665 / 0.670 | 0.680 / 0.691 |
  | 0.365 (central) | 0.655 / 0.657 | 0.670 / 0.674 |
  | 0.50 | 0.640 / 0.645 | 0.655 / 0.658 |

  Hull over the scan: **MAP ∈ [0.64, 0.68], mean_h ∈ [0.645, 0.691]; floor mass ≤ 5e-3
  everywhere** (vs 0.2208 off). The floor departure is decisive at every scanned ρ.

**Caveats (all disclosed in the registration draft):**
- C1 — class split by the iiib host map (76 in-cat); joint_r1's own in-catalogue count is 73
  (row #270 §1.5). 3-event mismatch, immaterial at the band scale, disclosed.
- C2 — the central modeling assumption: the score-based ρ (0.2604) is scaled by the
  survival-aggregate ratio R_K; survival-ρ (0.104) and score-ρ (0.260) are different
  functionals (2.5× apart), so proportional transfer is approximate — the band scan spans
  ρ ∈ [0.26, 0.50] to cover it, and the run itself measures the truth.
- C3 — extrapolation: the response is anchored at h ≈ 0.73, the destination is 0.65-0.67;
  the lin/quad spread (~0.015 in MAP) is the honest scale of this residual — the same
  caveat class BAND_REDERIVATION §2.3 carried for iiib.
- C4 — candidate-set composition: joint_r1's cones/mass-window select on OBSERVED mass
  (window retention on the scattered venue 0.78-0.83 at the production linear k=1.5,
  row #270 §2), so the joint_r1 candidate set is not the iiib set with smeared queries;
  this derivation models the S_4D-query smearing only. Direction unbounded here; the
  measurement closes it. (Couples to Branch F d-t5-window: a window change after
  registration forces a revision.)
- C5 — ~25-28% of GH weight at true hosts falls outside the observed lnM range of its
  z-slab and is flat-extended (`truehost_clamp_w_mean`); this affects the expectation form
  (0.951) far more than the realized median (plateau-side draws dominate the median), and
  the realized-median is the registered object.

## 6. Physics-change status

No trigger file was touched; the flag, the factor, and all sites already exist in
production (rows #282/#286). This node concludes **no trigger-file change is required** for
Branch C: the registered measurement runs the existing post-flip default on the existing
joint_r1 configuration. (Had the derivation shown the "kernel" mass-form must be engaged on
the scattered venue, that WOULD be a physics change; it does not — the point form is the
production convention and the venue difference lives in the data, not the code. The
point-vs-kernel question remains the separately-tracked, already-bounded item of row #270
§1.3.)

## 7. Summary of derived values (for d-jr1-band)

- **T_jr1 (realized-median) = 1.031, h-spread 4e-4 (1.0316/1.0314/1.0312), 95% MC band
  [1.021, 1.036].** iiib comparand 1.039 (row #282). σ→0 limit recovers the delta law. PASS
  on all registered requirements (limiting case, h-stability, g-invariance, dimensions).
- **R_K (dark smearing ratio) = 1.40**; central ρ_jr1 = 0.365.
- **Predicted joint_r1 on-arm: MAP 0.655-0.670, mean 0.657-0.674 (central); scan hull MAP
  [0.64, 0.68], mean [0.645, 0.691]; floor mass ≤ 5e-3.**
- Proposed registered band (see REGISTRATION_DRAFT.md): **map_h AND mean_h ∈ [0.64, 0.70]**.
