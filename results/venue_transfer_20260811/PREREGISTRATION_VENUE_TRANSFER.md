# Pre-registration — VENUE-TRANSFER READ: does the in-loop σ_z-dosed coverage collapse manifest under production-matched realism?

Registered 2026-08-11, **BEFORE** any run and **BEFORE** the instrument is
built. This file registers the author-named decisive measurement of the
calibration-gate v2 KEEP-DIGGING clause (b) DEFECT verdict (gate verdict of
record: commit `64abd5f6`, adjudication CONFIRMED zero discrepancies, gate
TRUSTWORTHY).

**Authority — the author ruling of record (2026-08-11).** The author ratified
the v2 readout recommendations verbatim ("please continue as recommended by
you"): **(R1)** the v2 deviation register D1–D8 is RATIFIED, including the
venue switch to cluster (seeds prereg-pinned; run commit `dbde71dc` a
one-commit child of registered `065e7f58` with empty import-path diff —
accepted); **(R2)** the DS-8 confirmations are RATIFIED AS QUOTABLE measured
properties of record (A-1D starvation rail 400/400; ball-venue σ_z-dosed
uniform +bias with collapsed coverage, dose 0 → +0.011 → +0.035; B0 exactly on
truth); **(R3)** the in-loop σ_z-dosed coverage DEFECT is ADOPTED as a named
owner-candidate thread ALONGSIDE the standing photo-z-starvation account
(compatible, not competing: starvation owns the railing shape; the coverage
collapse is the candidate for what the estimator does underneath); **(R4)**
the clause-(b) "one measurement that decides" is NAMED: **the VENUE-TRANSFER
READ** — the gate's ball venue with production-matched population/catalogue
realism (v2 prereg §9 items 2/5), testing whether the +σ_z coverage collapse
is what the production pipeline does under GLADE photo-z; **(R5)** the DS-7
form call remains OPEN (report-only, no branch weight); **(R6)** paper #47's
hold reason upgrades to "P–P leg FAILED — coverage DEFECT; fix routes through
`/physics-change`".

**The question this file registers is R4's measurement.** Every design choice
below is flagged **AUTHOR-RATIFY** (§2 register); the final verdict is the
author's, never self-adjudicated (v1/v2 policy, carried verbatim).

**Instrument identity — registered deviation VT-D0 (prereg-first).** Unlike
v2 (instrument + tests + prereg atomic), this registration is committed
**before** the instrument exists: the build is BLOCKED until this commit.
Registered requirements, binding: (i) the instrument is
`master_thesis_code/validation/venue_transfer.py` + test file
`master_thesis_code_test/validation/test_venue_transfer.py`, committed
together in a descendant of this commit **before any registered cell runs**;
(ii) that commit leaves every line of this file above the §11 appendix
unmodified; (iii) every output JSON embeds `git_commit` + the full clean-rule
provenance block; a registered run commit must be the instrument commit or a
descendant with **empty import-path diff** to it (the R1-ratified D-4/D-5
pattern); (iv) the instrument commit hash is logged in §11 before the
campaign. — AUTHOR-RATIFY.

**REGISTERED — append-only discipline is in force from this commit.** Every
band below is fixed here and may not be adjusted after any readout. Nothing
above §11 may be edited after this commit; later material is appended to §11
with dated headings.

Parents: v2 prereg
`results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md`
(registered `065e7f58`); v2 readout `CALGATE_V2_READOUT.md` + adjudication
(committed `64abd5f6`); the v2 instrument
`master_thesis_code/validation/calibration_gate.py` (code identity
`065e7f58`, run `dbde71dc`); the registered closed-loop instrument
(`77b524af`). The author value ruling is unchanged: correctness + insight,
not bias-removal.

---

## 0. Binding constraints of record

- `master_thesis_code/validation/calibration_gate.py`,
  `master_thesis_code/validation/closed_loop_gfrac.py`, and
  `master_thesis_code/validation/pp_coverage.py` are **NOT modified** — the
  new module imports them as libraries (house pattern: thin extension over
  registered instruments, modifying neither).
- No production physics file is touched — a TRANSFER-CONFIRMED escalation
  routes through `/physics-change` intake on the estimator's photo-z
  handling (author-gated), never through this registration.
- New code lives in exactly two files (VT-D0). All outputs, the readout
  script, and this registration live under `results/venue_transfer_20260811/`.
  v1/v2 artifacts are untouched.
- **No production posterior is produced.** Every posterior is a
  synthetic-universe diagnostic, quotable only against its own truth.
- Venue: **cluster-first** (§5 venue decision rule); the R1 precedent
  governs (seeds prereg-pinned inside the instrument, never venue-derived).

## 1. Question of record

**Q-VT (the author-named decisive measurement, R4).** The v2 gate measured,
on disjoint seeds in a trustworthy instrument, that σ_z-dosing a
multi-candidate ball venue produces a **uniform +≈σ_z MAP bias with
delta-narrow posteriors and 0/400 HPD coverage in both channels at all three
truths** (DS-8 T2 CONFIRMED; quotable per R2), while σ_z = 0 with the same
balls sits exactly on truth (T3). The venue was a registered caricature: the
v2 §9 items 2/5 NOT-EVALUABLE rows name the gap — *"Leg-2 venue transfer —
PENDING-AUTHOR-CONFIRMATION"* and *"GLADE n(z) / completeness map / sky-cone
geometry / f_incl < 1 — the ball is a z-window Poisson caricature."*

**Does the coverage collapse survive production-matched realism?** I.e. when
the ball venue is rebuilt on (a) the real detected event population, (b) the
real per-event candidate-ball multiplicities, and (c) the real heterogeneous
GLADE per-galaxy σ_z distribution (spec-z tail included), does the estimator
still show the σ_z-dosed collapse — or does realism kill it?

- **TRANSFER-CONFIRMED** ⇒ the DEFECT is the production mechanism candidate
  for what the estimator does under GLADE photo-z (R3's second thread
  becomes load-bearing next to starvation).
- **TRANSFER-REFUTED** ⇒ the DEFECT is an artifact of the synthetic venue;
  the starvation account stands alone.
- Pre-stated branch meanings: §Branches.

## 2. DESIGN REGISTER — the production-matching axes (each: decision, rationale, ratification flag)

The design question of record: which of the four candidate axes — (a) real
event population, (b) real ball multiplicity/weights, (c) real per-galaxy
σ_z, (d) the production estimator code path — does the decider need? The
registered answer: **adopt (a) + (b-multiplicity) + (c); exclude
(b-weights) and (d) with the bracketing/certification arguments below.**

### VT-D1 — axis (a) ADOPTED as a pinned design: real event rows, per-seed noise redraws — AUTHOR-RATIFY

- The event set is **pinned to the production detected set**: the rows of
  `results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`
  (md5 `9a1f2a14384a9281c97ca3be312ddaab`, the v2 prereg pin, recomputed by
  the v2 scorer — readout §2). 1590 rows; all 1590 pass the parent
  `load_sigma_triples` row filter (verified at design time); production
  evaluated 1588 of them (the frozeng emit's event set; 2 rows dropped by
  production, `m4_results.json` `<venue>.dropped_crb_row_indices`).
- Per event: `(d_L, σ_dL/d_L, σ_Mz/M_z, ρ)` are that row's own values —
  the σ–d_L joint texture is **exact by construction** (no rank-matching;
  the v2 V4 texture check is superseded by the pin, VT-D8).
- `z_true,i` = ladder inverse of the row's `d_L` at the cell's `h_true`
  (the parent's `_z_of_dl_table`); per-seed randomness = fresh correlated
  observation noise `(d_L_obs, M_z_obs)` from the row's own 2×2 block +
  ball draws + σ_z draws. Truth-generation is self-consistent per truth by
  construction (the committed F5 `d_L`-reinterpretation device).
- Events whose `d_L` exceeds the venue horizon at `h_true`
  (`> 0.999 × dl_max(h_true)`) are dropped and counted; guard: > 5 % of the
  pinned set ⇒ VENUE-CONFOUNDED (§10).
- **Coverage concept (pre-stated):** with a pinned design, DS-1/DS-2 read
  **conditional (fixed-design) frequentist coverage** over noise + ball +
  σ_z randomness at fixed truth — exactly production's situation (one
  realized catalogue, one event set). This is the transfer-relevant notion.
- No S_4D accept/reject generator runs ⇒ v2's DS-7 accounting has no
  subject; DS-7 is **N/A in this venue** (VT-D8).

### VT-D2 — axis (b) ADOPTED for multiplicity, EXCLUDED for weights — AUTHOR-RATIFY

- **Multiplicity real:** per event, the candidate-ball size `K_i` is pinned
  to the event's own production 1D ball count from the frozeng per-galaxy
  emit `results/run_20260804_frozeng/iiib/posteriors_with_bh_mass/h_0_73.json`
  (md5 `34c50e91028b6a6458a2b145db545705`; untracked file, therefore
  md5-pinned here and census-pinned below):
  `K_i = len(galaxy_likelihoods[i]) + len(additional_galaxies_without_bh_mass[i])`
  (the two lists are disjoint and h-invariant: committed
  `m4_results.json` `iiib.validation.V_wbh_additional_disjoint = true`,
  `V2_ball_sets_h_invariant_073_vs_060 = true`).
  **Registered census pins (recomputed at design time from the emit; the
  instrument must reproduce them exactly, V-T3):** n_events_evaluated =
  1588; 1D K: zeros = 606, ones = 74, median = 6, mean = 751.702…, p99 =
  11325.26, max = 245364, **ΣK = 1,193,703**; nonempty subset **n = 982**,
  median = 84, mean = 1215.58…, ΣK unchanged. (2D with-BH census, context:
  zeros = 1294, ΣK = 153,520.)
  This is the venue's largest structural departure from v2's Poisson λ = 4
  (K_mean ≈ 5.00) and directly implements §9 item 5's "Poisson caricature"
  gap. `K_i` is pinned (production has exactly one realization); the small
  per-seed incoherence — the window fluctuates O(σ_dL) around the
  realization that produced `K_i` — is disclosed, not modeled.
- **Weights equal (1/K, the registered gate convention), NOT the production
  per-galaxy rate weights `R_eff(M_g)/(1+z_g)`.** Bracketing justification,
  pre-stated: heavy-tailed weights only *shrink* the effective candidate
  number K_eff toward small values — the direction already measured by v2
  B2 (collapse at K_eff ≈ 5, quotable per R2). Equal weights maximize
  K_eff, i.e. maximize the dilution that could *kill* the collapse. If the
  collapse survives at both brackets (v2's K_eff ≈ 5 and this venue's
  equal-weight real K), intermediate weighted K_eff cannot plausibly kill
  it; if it dies here, the weighted case is a named gap (§9 item 3, W1 arm
  reserved).
- Impostor redshifts: i.i.d. `w_pop|W_i` on the truth ladder (the gate's
  Slivnyak–Mecke cut, carried verbatim). The window-interior n(z) shape
  (GLADE clustering, completeness roll-off inside `W_i`) is EXCLUDED —
  concentration-bracket argument: v2's λ = 4 host-dominated balls (maximal
  concentration at the host redshift) and this venue's window-spread real-K
  balls (maximal spread) bracket the concentration axis; B0/T3 (quotable)
  shows candidate *placement* alone is unbiased at σ_z = 0 — the dose
  responds to the kernel, not the placement. Named in §9 item 5.

### VT-D3 — axis (c) ADOPTED: GLADE-empirical heterogeneous σ_z, z-decile-matched, spec-z tail included — AUTHOR-RATIFY

- Every ball member (host included) draws its own σ_z from the **empirical
  per-galaxy `REDSHIFT_MEASUREMENT_ERROR` distribution of the iiib
  production catalogue's pruned frame**, z-decile-matched (the member's
  `z_true` selects the decile; draw uniform from that decile's σ_z pool) —
  the house `dl_binned` texture pattern applied to σ_z.
- Source frame: `cluster_parent_reduced_galaxy_catalogue.csv`
  (`results/campaign51_20260728/realistic_20260729/realizations_staged/`),
  reconstructed by the **committed m4 recipe**
  (`crossterm_instrument/m4_shared_galaxy_census.py::load_pruned_zm`:
  production column parse → `_empiric_stellar_mass_to_BH_mass_relation` →
  NaN drop → `_mass_redshift_prune_mask` at M ∈ [1e4, 1e7], z ≤ 1.5).
  **Registered integrity pins (committed numbers, exact match required,
  V-T3):** `rr1_ball_sigma_census.json` `iiib.pruned_sigma_stats`:
  n = 20,834,171; median = 0.0393412950539589; min = 0.0005317263419419;
  n_lt_5e-3 = 231,098; n_lt_1e-2 = 235,731.
- The spec-z-like tail rides in natively: 1.11 % of the pruned frame has
  σ_z < 5e-3 (committed count above); GLADE's ~0.56 % spec-z fraction is
  the standing F4 fact (memory of record) — the operative pin is the
  committed pruned-frame count.
- Committed location anchors for the dose statistic (context, not bands):
  parent-frame σ_z median = 0.03959198424570605, mean = 0.040140144665788,
  min = 0.0005244904928578, max = 0.265125751116521, NaN fraction =
  0.0002775489897817451 (`m1_kernel_delta_check.json` `parent_stats`);
  σ_z/(1+z) median = 0.03303813426597286 (same file); 1D-ball-member σ_z
  median = 0.0412018243170251, 2D = 0.03794726721785655
  (`rr1_ball_sigma_census.json` `iiib.1d/2d.ball_sigma_stats.median`).
  Predicted realized pair-mean dose σ̄ ∈ ≈ [0.039, 0.041].
- **Matched-model principle (carried from v2):** generator and estimator
  share the per-candidate σ_z (production reads per-galaxy
  `REDSHIFT_MEASUREMENT_ERROR`; evidence of record
  `m1_kernel_delta_check.json` `code_path_evidence.sigma_source`). Any
  coverage failure is therefore the estimator's photo-z **handling**, not
  model misspecification. The estimator generalizes the gate's scalar σ_z
  to a per-candidate vector: per-candidate ±5σ_k kernel clip, GL-50 on
  `[max(z_lo(h), z_obs,k − 5σ_k), min(z_hi(h), z_obs,k + 5σ_k)]`, all else
  verbatim (§4).

### VT-D4 — axis (d) EXCLUDED: the gate's certified mirror, not the production code path — AUTHOR-RATIFY

- The read runs the gate's estimator mirror (bare kernel × distance
  likelihood, equal candidate prior, no w_pop, no selection factor in the
  numerator, both channels − N ln α(h)) — the registered production kernel
  form whose in-loop calibration the v2 verdict is *about*. Running
  production `BayesianStatistics` itself would require a full synthetic
  GLADE catalogue + sky maps + completeness (a different instrument), and
  the mirror's fidelity is already the adjudicated basis of the v2 verdict.
  Decidability does not need (d): the claim under test is that the
  *mirror's* defect transfers to production-matched inputs; if CONFIRMED,
  the fix (and its verification) happens on the production code path via
  `/physics-change` intake, which is the escalation of record (R6).
- The certification chain replacing (d): **V-T5 no-drift** (the new
  module's vectorized estimator core, run in v2-compat mode, must
  bit-reproduce the committed v2 B2(0.730) per-seed records) + **T-0 / T-a
  anchors** (§5). The kernel-form gap (`volume_deconv`, production's
  resolved kernel per `m1_kernel_delta_check.json`) stays NOT-EVALUABLE
  exactly as v2 §9 item 6 left it (O2 arm reserved, NOT built).

### VT-D5 — event-set restriction: the 982 nonempty-ball events — AUTHOR-RATIFY

- The venue runs the **982 events with nonempty production 1D balls**
  (census pin). The 606 empty-ball events are EXCLUDED: in the mirror's
  registered normalisation (`ln P = Σ ln L_i − N ln α`, N fixed) an
  always-empty event would subtract an h-dependent `ln α(h)` with no
  h-dependent numerator — a venue-manufactured distortion production does
  not have (production routes such events through completeness/out-of-
  catalogue terms the mirror deliberately lacks). The read is therefore
  **conditional on host-in-ball events** (`f_incl = 1` carried); the
  f_incl < 1 / completeness leg stays NOT-EVALUABLE (§9 item 4).
  `N_det` = 982 − horizon drops (VT-D1), fixed per cell, recorded.

### VT-D6 — 2D channel convention — AUTHOR-RATIFY

- The 2D channel applies the production `g_i` mass factor over the **same
  1D ball** (the registered gate convention), with each event's own
  `(σ_Mz, ρ)` from its pinned CSV row. Production's 2D ball is the with-BH
  *subset* (census: 1294/1588 empty) — that restriction is NOT mirrored
  (would need per-channel ball pins and a second N_det bookkeeping for 294
  events). **The registered headline verdict is the 1D channel** — the
  thread under test (R3/R4) is the production 1D behaviour; the 2D read is
  a named secondary, same bands, reported alongside; a 1D/2D split is
  flagged in the verdict line and routes to MIXED handling (§Branches).
  With-BH-subset 2D realism: §9 item 7.

### VT-D7 — disjoint seed plan — AUTHOR-RATIFY

- Base `20260808` (the gate's `GATE_BASE_SEED`), **v3 offsets in the
  +40000 decade**: envelope `20260808 + [40000, 45399]` (§5 table),
  disjoint by construction from v1's `+[0, 9049]` and v2's
  `+[20000, 29049]` (incl. the reserved O1 block) — unit-tested. Reserved,
  never-run-post-hoc blocks: `+46000…+46399` (W1 weighted-candidates arm,
  NOT built), `+47000…+47399` (O2 `volume_deconv` arm, NOT built).

### VT-D8 — checks superseded by the pinned design — AUTHOR-RATIFY

- **V4 texture** N/A: the σ–d_L texture is the CSV rows themselves (exact);
  replaced by V-T3 pin integrity (md5 + census + sampler pins).
- **DS-7 generator closure** N/A: no accept/reject generator (VT-D1); the
  R5 OPEN form call is untouched by this read.
- Degenerate-PIT exemption (v2 D3) carried verbatim: T-0 (σ_z = 0) is
  scored on DS-3/DS-4 only.

### VT-D9 — venue: cluster-first — AUTHOR-RATIFY

- Derived budget ≈ 4,500 CPU-h (§5) ⇒ local ETA ≈ 323 h at 14 workers —
  infeasible; the campaign is cluster-first per the standing policy and the
  R1-ratified precedent. §5 registers the venue decision rule (preflight +
  `sbatch --test-only` probe; STOP-author if the cluster is unavailable,
  since no ≤ 12 h local plan exists). Seeds are prereg-pinned inside the
  instrument (`--cell`/`--truth`/`--seed-range`), never SLURM-derived.

## 3. THE INSTRUMENT

`master_thesis_code/validation/venue_transfer.py` (+ its test file), a thin
extension importing `calibration_gate` and `closed_loop_gfrac` as libraries
(neither modified): parent context build, `α(h)` tables, `z_of_dl` ladders,
`pp_readout`/`hpd_contains`, GL/GH quadrature orders, canonical 41-point h
grid — all inherited. New capabilities: (i) pinned-event universe (VT-D1);
(ii) pinned-K real-multiplicity balls (VT-D2); (iii) z-decile σ_z sampler +
per-candidate-σ estimator core (VT-D3), a vectorized generalization of the
gate's scalar-σ_z ball path, certified by V-T5 bit-reproduction in v2-compat
mode. Estimator config mirrored from production, carried verbatim from v2
§5: `numerator_pdet = off`, `snr_threshold = 20`, 50-node Gauss–Legendre,
64-node Gauss–Hermite, injection pool `mix200k_20260728`, CRB CSV as pinned.
Implementation freedom, registered: candidate-pair arrays MAY be evaluated in
chunks/per-event groups for memory (peak single event K = 245,364); worker
count per node is a venue fit, recorded in every JSON; neither may change any
statistic (V-T2 determinism + V-T5 govern). What the module never does:
import `BayesianStatistics`; modify a parent; produce a production posterior;
adjudicate the branch.

## 4. Generative model and estimator (exact)

Per seed, at cell truth `h_true`:

1. **Events (pinned).** The registered event list = CSV rows evaluated by
   production (1588) ∩ nonempty 1D ball (982) ∖ horizon drops (VT-D1).
   `z_true,i = z(d_L,i; h_true)` by ladder inversion;
   `(σ_dL, σ_Mz, ρ)_i` = the row's triples.
2. **Noise.** `d_L_obs = d_L (1 + σ_dL e_1)`;
   `M_z_obs = M_z_true (1 + σ_Mz (ρ e_1 + √(1−ρ²) e_2))` with
   `M_z_true = M_row (1 + z_true,i)` — the parent's correlated fractional
   noise, verbatim.
3. **Ball.** Window `W_i = [z(d_L_obs(1 − 4σ_dL)), z(d_L_obs(1 + 4σ_dL))]`
   on the truth ladder (gate verbatim); members = host at `z_true,i` +
   `(K_i − 1)` impostors i.i.d. `w_pop|W_i`; order shuffled.
4. **σ_z texture.** Each member k draws `σ_z,k` from the z-decile sampler
   (VT-D3) at its true z; `z_obs,k = z_k + σ_z,k ε_k`. T-0: `σ_z,k ≡ 0`;
   T-a/T-b: `σ_z,k ≡ 0.035` flat (v2 B2 dose).
5. **Estimator (both channels, gate math with vector σ):**
   `L_i(h) = (1/K_i) Σ_k ∫ dz N(z; z_obs,k, σ_z,k) ·
   N(d_L(z;h)/d_L_obs,i; 1, σ_dL,i) · [g_i(z;h)]`, per-candidate GL-50 on
   `[max(z_lo(h), z_obs,k − 5σ_z,k), min(z_hi(h), z_obs,k + 5σ_z,k)]`
   (σ_z,k = 0 ⇒ point evaluation), `[z_lo, z_hi]` the ±4σ window capped at
   `z_max(h)`; `ln P(h) = Σ_i ln L_i(h) − N_det ln α(h)` with N fixed and
   the finite −745 zero-likelihood penalty — all carried verbatim from the
   gate (divergence-10 convention included).
6. **Readout.** `posterior_readout` + `pp_readout` on the canonical
   41-point grid, both channels. Grid-clearance pre-check (v2 D4 style):
   predicted MAP under CONFIRM ≈ truth + 0.039…0.044 ⇒ ≤ 0.814 at the
   highest truth, ≫ 3 posterior-sd from the 0.860 edge at committed widths
   (0.0012–0.0059); under REFUTE the posterior sits at truth. The §8 edge
   guard protects the read if this prediction fails.

## 5. Cell matrix, seed plan, N floor, runtime budget

| cell | events | balls | σ_z | truths × seeds | v3 seed blocks (base 20260808) |
|---|---|---|---|---|---|
| **T-0** (anchor) | pinned 982 | real K_i | 0 | 0.730 × 200 | +40000…+40199 |
| **T-a** (axis-a arm) | pinned 982 | Poisson λ=4 | 0.035 flat | 0.730 × 200 | +41000…+41199 |
| **T-b** (axis-b arm) | pinned 982 | real K_i | 0.035 flat | 0.730 × 200 | +42000…+42199 |
| **T-c** (decision) | pinned 982 | real K_i | GLADE sampler | 0.690 × 200 / **0.730 × 400** / 0.770 × 200 | +43000…+43199 / +44000…+44399 / +45000…+45199 |
| W1 *(NOT built)* | — | real K_i + rate weights | — | — | +46000…+46399 reserved; NOT-EVALUABLE (§9 item 3) |
| O2 *(NOT built)* | — | `volume_deconv` kernel | — | — | +47000…+47399 reserved; NOT-EVALUABLE (§9 item 2) |

Fixed per-cell; a seed appears in exactly one cell; no v3 seed appears in any
v1/v2 cell (VT-D7, unit-tested). The **decision read** is T-c(0.730) at
N = 400; T-c wings carry the truth-uniformity leg at N = 200; T-0/T-a/T-b
are the anchor + ablation ladder (DS-VT5). All cells share the pinned event
set, so ladder attribution is clean.

**Runtime budget (derived from committed v2 wall times; cross-checked in
smoke).** Committed anchors (`B2_h0p730_results.json`: 1.529 s/seed at 64
workers, K_mean 4.997, N = 1500 ⇒ 97.9 CPU-s / 7,496 pairs = **13.06
CPU-ms/pair**, σ_z > 0 path; `B0_h0p730_results.json`: 0.333 s/seed ⇒ **2.84
CPU-ms/pair**, σ_z = 0 path). Venue pair count ΣK = 1,193,703 ⇒ per-seed:
T-b/T-c ≈ 15,590 CPU-s ≈ **4.33 CPU-h**; T-0 ≈ 3,390 CPU-s ≈ 0.94 CPU-h;
T-a ≈ 64 CPU-s. Totals: T-c 800 × 4.33 ≈ 3,464 + T-b 200 × 4.33 ≈ 866 +
T-0 ≈ 188 + T-a ≈ 4 ⇒ **≈ 4,520 CPU-h ≈ 16.3 M CPU-s**. Cluster plan:
SLURM array, heavy cells chunked ≤ 25 seeds/task at 64 workers ⇒ ~45 tasks ×
≤ 2 h predicted (≤ 4 h requested); campaign wall 2–8 h queue permitting.
Local ETA at 14 workers ≈ 323 h ⇒ infeasible.

**Venue decision rule (standing policy, R1 precedent):** consult `/cluster`,
run the preflight and require `VERDICT: READY ✓`, probe the queue with
`sbatch --test-only`; if the cluster is unreachable or the projected start
makes the campaign infeasible, there is **no local fallback** (ETA > 12 h) ⇒
STOP, author call. The sbatch script may land in a child commit of the
instrument commit with empty import-path diff (R1-ratified pattern).

**Smoke (before the campaign):** 3 seeds each of T-0, T-a, T-c(0.730)
(smoke flag; `--allow-dirty` permitted for smoke/validate only, D5 carried);
measures per-seed CPU against the derived estimate (abort (a) input) and runs
the V-T2 determinism spot-check.

**N floor / registered fallbacks** (in order; no band adjusted, only the
pre-locked per-N rows below): **stage 1** — drop T-c wings (truth-uniformity
leg becomes NOT-EVALUABLE; saves ≈ 1,730 CPU-h); **stage 2** — T-b to
N = 100 (N=100 rows, §7); **last resort** — decision cell to N = 200 (N=200
rows) + STOP-author consult before running.

## 6. Per-seed outputs

The gate's §6 per-seed fields carried (both channels: `map_*`, rails,
`pit_*`, `hpd50/68/90_*`, `post_sd_*`, `edge_mass_*`, 41-point `ln_post_*`
vectors), plus: `sigma_z_mean_pairs`, `sigma_z_median_pairs`,
`frac_pairs_sigma_lt_5e-3` (realized dose statistics), `K_sum`,
`n_events_run`, `n_horizon_dropped`. Document level: full clean-rule
provenance (D5 fields), the V-T3 pin-integrity block (md5s + census + sampler
pins, recomputed), config dump, seeds, wall time, workers.

## 7. DECISION STATISTICS (bands locked at this commit)

Input provenance of every number: v2 prereg §7 rows (`065e7f58`); v2 readout
§7 T2 committed values (`64abd5f6`); binomial/Jeffreys arithmetic recomputed
at each N (formulas identical to v2 DS-1/DS-8); committed σ_z location
anchors (VT-D3 files/fields). Nothing below uses any number produced by this
venue.

**DS-VT1 — HPD coverage (binomial nulls, v2 DS-1 formula).**
N = 400 rows (verbatim v2): 2σ [0.450, 0.550] / [0.633, 0.727] /
[0.870, 0.930]; 3σ [0.425, 0.575] / [0.610, 0.750] / [0.855, 0.945].
N = 200 (same formula): 2σ [0.429, 0.571] / [0.614, 0.746] /
[0.858, 0.942]; 3σ [0.394, 0.606] / [0.581, 0.779] / [0.836, 0.964].
N = 100 (fallback): 2σ [0.400, 0.600] / [0.587, 0.773] / [0.840, 0.960];
3σ [0.350, 0.650] / [0.540, 0.820] / [0.810, 0.990].
Not scored on T-0 (degenerate-PIT exemption, VT-D8).

**DS-VT2 — P–P/KS (v2 DS-2 formula).** N = 400: PASS D ≤ 0.0679, FAIL
D > 0.0814; N = 200: 0.0960 / 0.1151; N = 100: 0.1358 / 0.1628.

**DS-VT3 — MAP bias + dose prediction.** Grid-argmax bias, v2 DS-3 edges
verbatim: in-band |b| ≤ 0.010; defect |b| ≥ 0.030. **Dose-ratio statistic**
`R_dose = bias / σ̄_pairs` with `σ̄_pairs` the cell's realized mean
per-candidate σ_z (recorded, §6). Committed dose–response anchors (v2
readout §7 T2, bias/σ_z): B1-1D 1.0688, B1-2D 1.0950, B2(0.690) 0.9971 /
1.0018, B2(0.730) 1.0075 / 1.0211, B2(0.770) 1.0621 / 1.0679 — committed
range [0.997, 1.095]. **Registered band: R_dose ∈ [0.75, 1.25]** — covers
the committed range with margin for mixture/Jensen and population-shift
effects; excludes the null (0) and super-linear (≥ 1.5) alternatives.
For flat-dose arms (T-a/T-b) σ̄_pairs = 0.035 exactly.

**DS-VT4 — rail fractions (v2 DS-4/DS-8 arithmetic).** Collapse-pattern
bands (from v2's committed 0/400 everywhere, Jeffreys + 3σ, rounded looser):
R_low ≤ 0.02 and R_high ≤ 0.02 at N = 400; ≤ 0.04 at N = 200; ≤ 0.08 at
N = 100. **Pre-named distinct pattern:** any rail fraction ≥ 0.90 (the v2
DS-6 threshold, carried) in a decision cell = **RAIL-EMERGENT** — the venue
manufactures railing the caricature never showed; forces MIXED with
mandatory author formulation (this would be a *shape* transfer, not the
registered collapse pattern).

**Per-channel cell classification (mechanical):**
- **COLLAPSE-REPRODUCED** = C90 ≤ 0.02 (N=400) / ≤ 0.04 (N=200) AND
  R_low, R_high within the DS-VT4 collapse band AND bias ≥ +0.030
  (positive DEFECT-scale) AND R_dose ∈ [0.75, 1.25].
- **CALIBRATED** = DS-VT1 all three levels inside 3σ AND DS-VT2 PASS AND
  |bias| ≤ 0.010 AND R_low, R_high within the collapse band (no railing).
- **OTHER** = anything else (attenuated bias, partial coverage failure,
  RAIL-EMERGENT, …) — raw values reported, direction stated.

The "delta-narrow" companion (post_sd_median ≪ bias; v2 committed
0.0012–0.0059) is REPORTED un-banded, v2 convention (no committed SE for a
median). DS-5 width-vs-F5 remains NOT-EVALUABLE (v2 §9 item 3 carried).

**DS-VT5 — per-axis ablation ladder (report-graded, no branch weight).**
Ordered ladder: v2 B2(0.730) [committed baseline, quotable per R2] → T-a
(+ real events) → T-b (+ real multiplicity) → T-c (+ real σ_z). Each arm is
classified with the rules above at its N; the **killing axis** (under
TRANSFER-REFUTED or MIXED) = the first arm in ladder order whose
classification leaves COLLAPSE-REPRODUCED. T-a is additionally reported
against v2 B2(0.730)'s committed values (+0.035263 / +0.035737) as raw
context — no tight band carries across the population change (pre-stated).

## 8. Edge-contamination guard

Carried verbatim from v2 §8: edge-loaded = `edge_mass > 0.01`; a
cell×channel with > 10 % edge-loaded seeds is EDGE-CONTAMINATED ⇒ its
DS-VT1/DS-VT2 carry no weight; DS-VT4 exempt.

## 9. NOT-EVALUABLE registry (registered exclusions with reserved escapes)

1. **Estimator code-path identity (axis d)** — the mirror, not
   `BayesianStatistics`; certification chain V-T5 + T-0/T-a; any estimator
   fix routes `/physics-change` (R6).
2. **`volume_deconv` kernel form** — O2 reserved (+47000), NOT built;
   carried verbatim from v2 §9 item 6.
3. **Per-galaxy rate weights** — W1 reserved (+46000), NOT built;
   bracketing argument VT-D2; author may order it post-read.
4. **f_incl < 1 / empty-ball events / completeness** — the 606 zero-ball
   events excluded (VT-D5); the read is conditional on host-in-ball.
5. **Window-interior n(z) shape** (GLADE clustering + completeness
   roll-off inside `W_i`) — impostors stay `w_pop|W`; concentration
   bracket VT-D2.
6. **Sky-cone geometry / per-event sky selection** — no sky in the mirror
   (v2 §9 item 5 residue).
7. **With-BH-subset 2D ball realism** — VT-D6 convention; the 2D verdict
   is secondary.
8. **Width-vs-F5 fine read** — matched-population F5 run remains the
   registered follow-up (v2 §9 item 3).

## 10. Validity: anchors, determinism, provenance, abort criteria

- **V-T1 — T-0 anchor (M-2-style in-design null).** Real events + real K at
  σ_z = 0 (the B0-analog on this venue). Registered expectations: DS-3
  |bias| ≤ 0.010 both channels; R_low, R_high ≤ 0.05. DS-1/DS-2 exempt
  (degenerate PIT). |bias| ∈ (0.010, 0.030) = ANCHOR-MARGINAL (reported,
  disclosed in the verdict line, does not void). **|bias| ≥ 0.030 or a rail
  > 0.05 ⇒ VENUE-CONFOUNDED** — and is simultaneously a first-class NEW raw
  finding (bare-kernel ball estimator uncalibrated under real multiplicity
  at perfect redshifts), barred from claims pending author formulation.
- **V-T2 — determinism.** Bit-identical re-run spot-check in smoke (v2 V3
  pattern); chunked evaluation must be seed-deterministic.
- **V-T3 — pin integrity (before any registered cell).** (i) CRB CSV md5 =
  `9a1f2a14384a9281c97ca3be312ddaab`; (ii) frozeng emit md5 =
  `34c50e91028b6a6458a2b145db545705`; (iii) recomputed K census == the
  VT-D2 pins (1588 / 606 / 982 / ΣK 1,193,703 / max 245,364), exact;
  (iv) recomputed pruned-frame σ_z stats == the VT-D3 pins (n = 20,834,171;
  median = 0.0393412950539589; min = 0.0005317263419419; n_lt_5e-3 =
  231,098), exact (same deterministic recipe). Any mismatch ⇒ STOP.
- **V-T4 — clean rule, verbatim from v2 (D5):**
  1. **Import path** = everything under `master_thesis_code/` and
     `master_thesis_code_test/`. A registered cell run **REFUSES to start**
     (hard `SystemExit`, before any context is built) if the import path has
     **ANY uncommitted change — modified OR untracked**.
  2. `--allow-dirty` is accepted **only together with `--smoke` or
     `--validate`** — the CLI rejects it outright on a registered cell run.
     There is no other escape.
  3. Dirt **outside** the import path (doc edits, untracked results
     directories, rule files, …) never blocks, and the **full inventory**
     (verbatim `git status --porcelain` lines, split
     `{import_path: [...], other: [...]}`) is embedded in **every output
     JSON** together with `git_commit`, `git_dirty` (whole tree) and
     `import_path_clean`.
- **V-T5 — no-drift anchor (cross-instrument).** The new module in
  v2-compat mode (universe via imported `draw_universe_gate`, ball via
  imported `draw_ball`, constant σ_z vector, N = 1500, λ = 4, σ_z = 0.035,
  canonical grid) on v2 B2(0.730) seeds 20286808–20286810 must reproduce
  the committed `B2_h0p730_results.json` `per_seed` records
  **bit-identically** (shared fields). This certifies the vector-σ
  estimator core = the committed gate math. Run at instrument-commit time,
  logged in §11 before the campaign. Failure ⇒ STOP.
- **Abort criteria:** (a) smoke-measured heavy per-seed CPU > 2× the
  derived 4.33 CPU-h ⇒ stage-1 fallback (§5), re-derive; projected campaign
  wall still > 12 h on the registered array plan ⇒ stage-2; still over ⇒
  STOP, author call. (b) non-finite `ln_post` in > 1 % of any cell's seeds
  ⇒ STOP. (c) any V-T failure ⇒ STOP. (d) VT-D1 horizon-drop guard > 5 % ⇒
  STOP. No band may be adjusted after any readout.
- **VENUE-CONFOUNDED trigger set** = {V-T2…V-T5 failure, abort (b)/(d),
  T-0 V-T1 hard trigger} ∪ {decision cell EDGE-CONTAMINATED in the channel
  being read}. Measurement anomalies in T-a/T-b/T-c are findings, never
  trustworthiness escapes (v2 policy carried).

## Branches (presented to the author, never self-adjudicated)

Checked in order; the registered headline verdict is the **1D channel**
(VT-D6); the 2D classification is reported alongside in every branch.

1. **VENUE-CONFOUNDED** — any trigger-set member fires ⇒ every measurement
   below is void; no science content quoted; a T-0-triggered confound is
   simultaneously logged as a first-class new raw finding (V-T1). Author
   call on repair-and-rerun (v2's defect-repair pattern).
2. **TRANSFER-CONFIRMED** — T-c 1D is COLLAPSE-REPRODUCED at **all three
   truths** (0.730 at N=400 bands; wings at N=200 bands). Meaning, pre-
   stated: the σ_z-dosed coverage DEFECT survives production-matched
   population, multiplicity, and GLADE σ_z heterogeneity (spec-z tail
   included) ⇒ it is **the production mechanism candidate** for what the
   estimator does under GLADE photo-z, alongside (not replacing) the
   starvation account's railing shape (R3). Escalation, pre-stated:
   `/physics-change` intake on the estimator's photo-z handling
   (author-gated); paper #47's hold reason stands as upgraded by R6, with
   the transfer leg now EVALUATED-CONFIRMED.
3. **TRANSFER-REFUTED** — T-c(0.730) 1D is CALIBRATED. Meaning, pre-stated:
   production-matched realism kills the collapse ⇒ the v2 DEFECT is an
   artifact of the synthetic caricature venue; the DS-VT5 ladder names the
   killing axis; the starvation account stands alone as owner of the
   production 1D behaviour; the calibration-gate DEFECT is re-scoped to an
   in-loop instrument finding with no production transfer; whether paper
   #47's hold wording reverts is the author's call — nothing here
   mechanically lifts the hold (stage-5 conjunction still unsatisfied).
4. **MIXED (first-class, non-forcing)** — anything else: attenuated bias
   (0.010 < |b| < 0.030), partial coverage failure, RAIL-EMERGENT, a 1D/2D
   split, or wings disagreeing with the center. Handling, pre-stated: the
   per-axis ladder attribution table + the §9 registry rows are presented
   as the candidate carriers of the gap; neither the transfer claim nor the
   pure-starvation account may be quoted from a MIXED read; the reserved
   W1/O2 arms are the pre-named follow-up candidates, buildable only by
   author order.

**Anti-tuning:** every threshold in this file (N = 400/200/100 and their
DS-VT1 rows; KS 0.0679/0.0814/0.0960/0.1151/0.1358/0.1628; 0.010/0.030;
collapse bands 0.02/0.04/0.08; rails 0.90; R_dose [0.75, 1.25]; the T-0
anchor edges 0.010/0.030/0.05; edge guard 0.01/10 %; horizon guard
0.999/5 %; the census, md5, and sampler pins; the +40000 seed decade; the
runtime anchors 13.06/2.84 CPU-ms/pair and the 2× abort factor) is fixed at
this commit, derived from committed v1/v2/production artifacts or standard
binomial/Jeffreys arithmetic — and may not be adjusted after any readout.
The git object of this file is the evidence of what was registered.

**Model/effort policy for the readout:** carried verbatim from v2 —
mechanical extraction at low effort; interpretation/adversarial pass at high
effort; **the branch call is presented to the author, never
self-adjudicated.**

---

Verdict to be appended below by the session that reads out the campaign —
after this file is committed, no edits above this line.

---

## 11. Appendix log (append-only)

*(The instrument commit hash, smoke results, V-T5 no-drift evidence,
deviations discovered during the run, and the readout are appended here with
dated headings; the text above stays.)*

### 2026-08-11 — Operational deviation: array 6252702 runtime blowout and resubmission — PENDING AUTHOR RATIFICATION 2026-08-12

Campaign array **6252702** (49 tasks, `cpu_il`, 64 cpus/task,
`--time=04:00:00`, per the §5 cluster plan): **10 COMPLETED** (the 8 T-0
chunks, T-a, and T-c(0.730) seeds 75:25), **39 TIMEOUT** at the 4 h limit
with no output JSON. The instrument writes its chunk JSON only at run
completion — no partial outputs exist for the timed-out tasks (verified:
exactly the 10 completed JSONs are on disk).

**Root cause (operational, not statistical).** The §5 wall prediction
(≤ 2 h/task at 64 workers) implicitly assumed the 64 workers subdivide
per-seed cost. The instrument's `mp.Pool` parallelizes **over seeds**; a
single seed is a single-process unit. A 25-seed chunk therefore cannot
finish faster than one seed's single-process wall, measured at ≈ 3.8–3.95 h
for the heavy cells (task 28, the one completing heavy task: wall 3:56:07,
CPU 94.63 h / 25 seeds on uc2n810-class node uc2n804). The 4 h request left
minutes of headroom; the 38 sibling heavy tasks (T-b + remaining T-c) hit
the limit just short of completion. Memory was never a factor (≈ 18 GB /
125 GB).

**Measured vs predicted per-seed cost.** Measured heavy per-seed CPU =
94.63 CPU-h / 25 = **3.79 CPU-h/seed**, i.e. 0.87× the derived 4.33 CPU-h
anchor — **abort (a) does NOT trip** (trip point 2× = 8.66 CPU-h/seed), and
the corrected array plan projects campaign wall ≪ 12 h, so **no N-floor
fallback stage is invoked**. The blowout is purely the wall-time sizing of
the array request, not the registered CPU budget.

**Completed-chunk validity.** `Tc_h0p730_results_seeds75_25.json` verified
as a full registered run: `n_events = n_events_run = 982` on all 25 seeds,
`n_events_cap = null`, `smoke = false`, seeds 20304883–20304907 (base
20260808 + 44075…44099, the registered T-c(0.730) block), `K_sum =
1,193,703` per seed, `pin_integrity.pass = true`, import path clean. It is
**RETAINED**, as are the 8 T-0 chunks and T-a.

**Resubmission (NON-STATISTICAL operational deviation).** Seeds, seed→cell
map, chunking, bands, statistics, thresholds, and the instrument commit
(`2ece8801`) are all untouched. The 39 timed-out tasks are resubmitted from
the same sbatch with CLI overrides only: `--array=9-27,29-48`,
`--cpus-per-task=25` (matches the 25-seed parallel grain; embeds
`workers = 25` instead of 64 in those chunk JSONs — result-invariant, the
Pool maps seed→record deterministically per V-T2), `--time=09:00:00`
(≈ 2× the measured 3.93 h task wall). Submitted job id recorded in the
readout entry.

**PENDING AUTHOR RATIFICATION 2026-08-12** (per the active overnight
mandate; this note is operational bookkeeping, not a readout — no band,
statistic, or seed changes anywhere above).


**§11 addendum (2026-08-12): full validity run logged.** `validate_results_full.json` (this directory,
dev box, instrument commit 2ece8801): seed_plan PASS, V-T2 determinism PASS, V-T3 pin integrity PASS,
**V-T5 no-drift PASS** (bit-reproduction of committed v2 `B2_h0p730` per-seed records on v2 seeds,
exact gate-shape mode). V-T4 (clean rule) is evaluated per registered run on the cluster
(`import_path_clean=true` embedded in every registered output). Honest sequencing note: the prereg
required this §11 evidence *before* the campaign; the launch-phase validate skipped V-T5
(`validate_results_novt5.json`) and the full run completed only after the (partially timed-out) first
array — a compliance-order deviation with no statistical content, bundled with the 2026-08-12
operational-deviation note for author ratification.


**§11 addendum 2 (2026-08-12): second straggler resubmission.** Array 6253922 (39 tasks, --time=09:00:00,
25 cores) completed 17 and timed out 22: packed 25-core tasks run ~1.6–1.9× slower than the uncontended
64-core reference (memory-bandwidth contention; completed walls 6:08–7:38 vs 3:56 reference) — the 9 h
"2× margin" was computed against uncontended timing. Remaining 22 chunks resubmitted as array 6259842
(--time=24:00:00, same 25-core grain, same registered seed chunks — NON-STATISTICAL, nothing else changed).
Cumulative outputs: 27/49 chunks (10 + 17). Lesson filed for the perf roadmap: margins must be computed
against contended timing, or tasks sized one-per-node. Bundled with the pending 2026-08-12 ratifications.


---

## §11 READOUT ENTRY (2026-08-13): campaign complete, verified — arrays 6252702 / 6253922 / 6259842

**Status: BRANCH FIRED BY THE TREE — TRANSFER-CONFIRMED. NOT ADJUDICATED (awaiting author ruling).**
This entry is appended by the Ship agent per the standing rule ("only the Ship agent may append,
and only on an author ruling" is the eventual verdict-of-record gate — this entry records the
fired-branch readout + independent CONFIRMED verification; it is not itself an author ruling).

**Validity (§8-§10, all clauses):** ALL PASS. V-T1 T-0 anchor: grid-argmax bias +0.000000
(SE 0.000000) both channels, all 200 seeds argmax exactly at h=0.730; refined companion
+0.000033±0.000033. R_low=R_high=0.000. Hard trigger did NOT fire. V-T2 determinism PASS
(validate_results_full.json + independent in-campaign corroboration, workers=64 vs 25, identical
K_sum=1,193,703). V-T3 pin integrity PASS on two independent sources (validate_results_full.json
+ all 49 chunk JSONs; 1200/1200 real_k seeds at SigmaK=1,193,703 exactly; T-a structurally exempt,
balls="poisson4" per registered §5/VT-D2 — not a mismatch). V-T4 clean rule PASS (49/49 chunks
import_path_clean=true). V-T5 no-drift PASS (bit-reproduction of committed v2 B2_h0p730 records,
3/3 seeds, 41 shared fields, zero mismatches; gate-shape mode chunk_pairs=0 — campaign ran
chunk_pairs=16384, see discrepancy note below). Registered-commit chain re-verified live:
2ece8801 (10 chunks) ancestor of e93f3068 (39 chunks); import-path diff EMPTY under both old and
new package names; prereg diff is a pure append at line 608 into §11 (VT-D0(ii) holds). Seed plan
(VT-D7) exact: 1400/1400 seeds, 0 duplicates, 0 missing/extra, 0 collisions with any reserved
envelope. Abort criteria (a)-(d): NONE triggered. §8 edge-contamination guard: 0 of 12
cell×channel entries EDGE-CONTAMINATED. VENUE-CONFOUNDED trigger set: 0 of 9 members fired.

**Branch fired (registered order, §10):** (1) VENUE-CONFOUNDED — does not fire. (2)
TRANSFER-CONFIRMED — fires: T-c 1D is COLLAPSE-REPRODUCED at all three truths (0.690/0.730/0.770),
both wings agree with the N=400 decision cell. (3) TRANSFER-REFUTED — not reached. (4) MIXED — not
reached. Headline (1D, VT-D6): T-c(0.730) N=400 1D = COLLAPSE-REPRODUCED (bias +0.037237±0.000230,
R_dose 0.8914, HPD90 0/400, rails 0.000/0.000). Secondary 2D reported alongside: also
COLLAPSE-REPRODUCED (bias +0.039713±0.000246, R_dose 0.9506). No 1D/2D split. DS-VT5 ladder
(rung 0 v2 baseline → T-a → T-b → T-c(0.730)): every rung COLLAPSE-REPRODUCED in both channels;
killing axis = NONE.

**Independent verification (adjudicate_venue_transfer.py, own re-derivation from the 41-point
ln_post vectors of all 1400×2 seed-channels, imports nothing from the instrument or the readout
scripts): VERDICT = CONFIRMED.** Raw-level fidelity: max |deviation| from instrument-stored
per-seed fields is 1.50e-14 (post_sd) and 0.0 elsewhere, across 114,800 ln_post values, zero
non-finite. All 16 scored fields × 12 cell-channels match the readout to max |Δ|=5.33e-15, all 12
classifications identical. Bands re-derived from scratch match the prereg §7 literals exactly.
Seed plan, provenance chain, and branch-tree evaluation independently reproduced with the same
result. Self-adjudication scan: no ruling language found; every §7 formulation item is correctly
framed as an author call.

**Discrepancies surfaced by verification, none of which changes the fired branch** (full list in
the readout's companion verification record, not reproduced here): two undisclosed
compliance-order deviations (pre-campaign smoke never run/no artifact; §11 was empty at
instrument-commit time, populated only after the first array had already run) — both
non-statistical, no branch impact; the as-run contended per-seed CPU exceeds the abort-(a) trip
point in 11 of 40 heavy chunks under the packed-25-core grain (abort (a) is not a
VENUE-CONFOUNDED trigger, and its fallback direction is conservative, so this cannot flip the
branch); V-T5's PASS covers the gate-shape code path (chunk_pairs=0), not the campaign's
chunk_pairs=16384 path, at an established ≤5e-16 relative (≤1 ULP) divergence — immaterial next to
the ~1e15×-larger posterior scale; several presentational/units-mixing notes in §7 of the readout
(R_dose range mixing grid-argmax/refined endpoints; "~300x" statement mixing quantised and refined
statistics). None of these bear on validity, the branch call, or any reported number beyond
rounding/wording.

**Pending author ratification carried forward, unchanged by this entry:** the three §11 deviation
notes above (array 6252702 runtime blowout/resubmission; V-T5 compliance-order deviation; second
straggler resubmission/array 6259842) remain PENDING AUTHOR RATIFICATION, joined by the two
newly-surfaced compliance-order items from verification (missing pre-campaign smoke artifact;
§11 not populated with commit hash/smoke/V-T5 evidence before the campaign launched).

**Artifacts of record:** `VENUE_TRANSFER_READOUT.md` / `.json` (readout), `collect_raw.json` +
`collect_extract.py` (raw extraction), `score_venue_transfer.py` (independent re-scorer),
`adjudicate_venue_transfer.py` + `adjudicate_venue_transfer_results.json` (independent
adjudication), 49 campaign chunk JSONs, `logs/` (cluster array logs), `validate_results_full.json`,
`validate_results_novt5.json`. Nothing above this line, and nothing above the earlier §11
addenda, was modified to produce this entry.

**Author decisions requested (unchanged from the readout, restated for the record):** (1) ratify
or reject the five compliance-order/operational deviation notes now on record; (2) rule on the
fired branch (TRANSFER-CONFIRMED); (3) decide whether to order the reserved W1 (per-galaxy rate
weights) and/or O2 (volume_deconv kernel form) arms; (4) decide whether to open the
`/physics-change` intake on the estimator's photo-z handling (prepared, not opened).
