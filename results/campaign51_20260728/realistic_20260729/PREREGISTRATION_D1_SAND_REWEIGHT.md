# Pre-registration — D1: the S_and-consistent selection re-weight

Registered 2026-08-05, **BEFORE** the run. Research Cycle stage 2
(`docs/RESEARCH_CYCLE.md`); stage 0–1 parent:
`CLAIM_D1_P0WINDOW_20260805.md` (same directory, committed `751d7d98`).
Upstream gate that made this test necessary: gate **(vi) selection-function
consistency**, recorded **BLOCKING-for-ship / decision D1** in
`.planning/derivation-2dbias-fix-20260803/FIXB_PATHA_PACKAGE.md` §4, §8, and
`RUNBOOK_NEXT_SESSION_7.md` §1 item 2(a).

**REGISTERED — committed before submission. Append-only discipline is in force
from this commit.** Every band below was fixed at this commit and may not be
adjusted after any readout. The stage-2 STEP 0 measurement-before-gate read
(**B2**) ran *before* this file was committed and its result is folded in below
as a completed read (claim C10 of the parent) — the ordering is: bands
registered → B2 measured → B2 checked against its band → this file committed →
arms submitted. B3 is registered as still-open and is read out with the run.

---

## Binding constraint of record (RUNBOOK-7 §1.2b, verbatim)

> the existing 3135-event catalogue stays band-passed and must never be
> re-scored against band-blind objects; the p0-bounds retirement is
> simulation-side, for future campaigns only.

This registration is **compliant by direction**: it makes the estimator's
selection objects *band-AWARE* (`S_and`), matching the filter the pipeline
actually applied to the catalogue. It never removes the band-pass from one side.
The `ParameterSpace.p0` bounds are **not touched** by anything here.
**No production posterior is produced** — every posterior emitted is a
counterfactual diagnostic, quotable only against its own twin.

---

## Why this test is necessary (concrete provenance)

- The 3135-event catalogue of record was selected by `SNR ≥ 20 ∧ p0 ∈ [10.002,
  15.998]`; 8 345 / 12 039 = **69.3%** of SNR-passers were removed by the stale
  `ParameterSpace.p0` bound guard (`parameter_space.py:95-113`,
  `parameter_estimation.py:268-276`, `main.py:772-779`).
- Every selection object the inference builds (`Σ³ᴰ, Σ⁴ᴰ, β_G, β_Ḡ, D, p_det`)
  is built from `SNR ≥ 20` alone.
- The filter is **class-conditional**: `s_G/s_D = 0.2286246597604769 /
  0.3129747690740832 = 0.7304891075943567`
  (`fixb_x15_attribution/cand_b_joint_selection_results.json`).
- Gate (ii)'s −3.71σ closes to **−0.48σ** *only* under S_and scoring — i.e. the
  package already conditions a headline consistency number on a filter the
  estimator does not model (`FIXB_PATHA_PACKAGE.md` §2).
- The residual 2D displacement is carried by `g_frac(h) = B_num_wbh/B_num`
  (frozen-g CONFIRM, both venues: 0.780→0.660 and 0.800→0.640), and `B_num_wbh`
  is built from the with-BH-mass survival object — the very object D1 says is
  mis-specified (`gate_vii/PREREGISTRATION_FROZEN_GFRAC.md` VERDICT).

- **The class-conditional retention is h-DEPENDENT** — the cheap read that could
  have cancelled this run instead authorised it: `s_G/s_D` falls monotonically
  from 0.74291 (h = 0.60) through 0.73049 (h = 0.73, pin) to 0.71795 (h = 0.86),
  `Δ ln = −0.0342` = **6.84×** the registered 0.005 h-flat band
  (B2, `d1_b2_sand_hslope.json`; parent claim C10). An h-independent retention
  ratio cannot tilt a posterior; this one is not h-independent.

Nothing on either exoneration layer covers this (check recorded in the claim
file, §"Two-layer exoneration check").

---

## STEP 0 — measurement-before-gate (runs FIRST; may cancel the whole run)

Per hard rule 6 and amendment A1, two cheap reads run **before** any evaluate is
submitted. Their outcomes are registered here with STOP semantics.

| # | measurement | cost | STOP clause |
|---|---|---|---|
| **B2** | `cand_b_joint_selection.py`'s selection legs re-run over the **canonical 41-point h grid** (originally h = 0.73 only, `cand_b_joint_selection.py:64`), reporting `s_G(h)`, `s_D(h)` and `d ln(s_G/s_D)/dh` across the grid | 3 × grid build + 3 catalogue passes; no estimator run | If `\|Δ ln(s_G/s_D)\|` across the grid is **< 0.005** — i.e. the class-conditional retention is h-flat to 0.5 % — then D1 cannot tilt the mixture weight and the full re-weight is **downgraded to a bounded null**: report the bound, do not run the evaluates. |
| **B3** | analytic/one-pass check of whether `g_frac = B_num_wbh/B_num` depends on the with-BH survival object at all (C7's `Refute by:`) | reading `bayesian_statistics.py:3296-3331`, `:4474-4480` + one quadrature | If `g_frac` is invariant under `S_4D → S_and,4D` to ≤1e-6 per event, the C7 convergence route is **dead** and branch (b) below loses its principal mechanism. Record it; it does not by itself cancel the run. |

### B2 — RUN AND READ, 2026-08-05 (before this file was committed)

Instrument `d1_b2_sand_hslope.py`, output `d1_b2_sand_hslope.json`, both
committed with the parent claim (`751d7d98`); full statement = parent claim
**C10** [LOCAL]. Pin at h = 0.73 reproduced to **relative deviation 0.0** on
`s_G`, `s_D`, `s_G/s_D` and `dl_max` (`pin_reproduced: true`) — the instrument is
certified against `cand_b_joint_selection_results.json` before its sweep is used.

| h | `s_G/s_D` | `ln(s_G/s_D)` |
|---|---|---|
| 0.60 | 0.7429148117274993 | −0.29717389530944144 |
| 0.73 (pin) | 0.7304891075943567 | −0.31404095879323346 |
| 0.81 | 0.7227381549328574 | −0.3247082871056849 |
| 0.86 | 0.7179493704330814 | −0.33135622713734925 |

- `Δ ln(s_G/s_D)` over the grid = **−0.03418233182790781**
- `d ln(s_G/s_D)/dh` (endpoints) = **−0.13147050703041463**; finite-difference
  slope stays within [−0.13356, −0.12411] — **monotone, one-signed, no seam
  artefact** across all 41 points
- `dl_max_S = dl_max_and = 9.164987215485882` at **every** h (horizon invariance,
  the N1 null, already satisfied by the pools themselves)

**STOP-clause verdict: measured 0.0342 = 6.84 × the 0.005 band ⇒ the h-flat null
is FALSIFIED, the STOP clause does NOT fire, the three-arm run is authorised.**
(`h_flat_verdict_full_grid: false`, `h_flat_verdict_endpoints: false`.)

B2's result is the number stage 1 could not supply (the F5 engine is N/A —
unbiased by construction and mass-blind in its `p_det`; see the parent claim's
stage-1 section). It is the **scale** the bands below are now expressed against.

---

## The run

Three arms, one venue each for iiib and joint_r1 (6 evaluates total, 41 h each).

| item | value |
|---|---|
| Code commit | **`9a715405`** — `main` HEAD at registration, PINNED. (Drafting HEAD was `a7e0d559`.) |
| **Code change** | **NONE to production formulas.** The S_and objects are obtained by *substituting the injection pool*, not by editing the estimator: `SimulationDetectionProbability` builds `p_det` as the survival function of the horizon, so setting `SNR := 0` on p0-rejected injections yields exactly `P(d_hor ≥ d_L ∧ p0 ∈ W \| M_z)` on the same 60×40 grid, same estimator, same flags (`cand_b_joint_selection.py:1-21, 90-94`). |
| Minimal instrumentation (named) | **one standalone script**, `results/campaign51_20260728/realistic_20260729/d1_sand/make_sand_pools.py`, lifted verbatim from `cand_b_joint_selection.py:68-95`, which materialises the two derived pools and writes a sha256 manifest. Tagged **`instrumentation`** (plain GSD), **not** `formula` — it touches no file on the `/physics-change` trigger list. |
| Baseline pool (arm A0) | `results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728` — 707 files, 200 100 data rows, fingerprint `dist(1.3261748578964083, 0.73) = 9.164987 Gpc` |
| Composition-control pool (arm A1) | `pool_p0kept` — the **647** pool files that carry a `p0` column, unmodified rows (60 files / 6 000 rows / 1 426 SNR-passers from pre-plunge-window `code_rev a9f29e82` dropped). **Registered expected size: 647 files, 194 100 data rows** (= 200 100 − 6 000; `d1_b2_sand_hslope.json:pool_fingerprint.n_rows_kept_p0_present = 194100`). |
| S_and pool (arm A2) | `pool_p0window` — the same 647 files with `SNR := 0` wherever `p0 ∉ [10.002, 15.998]`. **Registered expected size: 647 files, 194 100 data rows, of which exactly 149 092 carry `SNR = 0`** (`d1_b2_sand_hslope.json:pool_fingerprint.n_rows_dropped_p0_reject = 149092`, 76.81 % of the kept rows). A1 and A2 must be **row-for-row aligned**: same files, same row order, differing only in the `SNR` column on those 149 092 rows. |
| CRB input | the existing `prepared_cramer_rao_bounds.csv`, symlink target `run_20260729_seed61000/` — the same 3 135-row file both post-fix runs consumed. **No re-simulation.** |
| Catalogues | unchanged per venue: iiib = idealized parent (sha256 `7af3f4f4…4bd7d9`, 20 834 171 pruned rows); joint_r1 = observed realization seed-900001 (sha256 `e8f7ab31…4f6751`, 19 874 547 pruned rows), staged under `realizations_staged/` |
| Estimator config | `NORMALIZATION_MODE=absolute_marginal`, `HOST_Z_KERNEL=volume_deconv`, `HOST_MASS_KERNEL=auto`, `pdet_z_resolved=True`, `pdet_wbh_z_resolved=False`, `dl_bins=60`, `mass_bins=40`, `estimator=local_linear`, `SNR_THRESHOLD=20` — the post-fix path-(A) pairing, unchanged |
| h grid | canonical 41 points: 0.01 on [0.60, 0.65] ∪ [0.79, 0.86], 0.005 on [0.655, 0.79] |
| Twins of record | `results/run_20260804_postfix/{iiib,joint_r1}/` — **arm A0 already exists and is NOT re-run.** A0 was produced at commit `658c428a` (`$WS/run_20260804_postfix_{iiib,joint_r1}/run_metadata_0.json`, read at registration), the arms at the pinned `9a715405`. The whole range `658c428a..9a715405` touches `master_thesis_code/` in exactly three commits: `121f57d8` and `07904540` (the `--freeze_g_frac_ref_h` and `--selection_in_completion_numerator` toggles, both **opt-in and documented byte-identical at their defaults** `None` / `"off"`) and `77b524af` (a **new** file, `validation/closed_loop_gfrac.py`, imported by nothing in the evaluate path). Both toggles are left at default in all four arms — see the flag-verification evidence recorded with the submission. Therefore A0 is code-comparable and **re-running it is not authorised by this registration.** |

### Design matrix

| arm | pool | what it isolates |
|---|---|---|
| **A0** | full 707-file pool | the production baseline of record |
| **A1** | `pool_p0kept` (647 files) | **pool-composition control** — the 3.0% of injections dropped for lacking a `p0` column. Isolates "the pool changed" from "the selection changed". |
| **A2** | `pool_p0window` (647 files, `SNR := 0` outside W) | the **S_and-consistent selection**: every object (`p_det`, `Σ⁴ᴰ`, `β_G^φ`, `β_Ḡ^φ`, `D̃^φ`, `w̃_G`, `B_num`, `B_num_wbh`) rebuilt band-aware |

**The decisive contrast is A2 vs A1.** A2 vs A0 confounds D1 with the pool
composition change and must never be quoted as the D1 effect. A1 vs A0 is a
registered near-null (below).

---

## Pre-registered readings

**Scoring rule (amendment A2, and the gate-(vii) interpretation revision):** the
verdict is read off the **per-event tilt DISTRIBUTION** and the **STRATUM
decomposition**. **Σ alone is uninformative**: a Σ moving by ≫10% is *not*
sufficient for any branch, and a Σ agreeing to within a few % is *not* evidence
of a null. Both the paired distribution and the stratum split must be reported
before any branch is declared.

### Definitions (fixed here, computed mechanically)

- Per-event channel-difference tilt, the gate-(vii) object:
  `Δ_e ≡ ln(L_cat_with_bh/L_cat_no_bh)@h=0.81 − same@h=0.73`
  (`gate_vii/paired_check.py`, unchanged code).
- Per-event **mixture** tilt, the posterior-relevant object:
  `t_e ≡ ln(combined_with_bh)@0.81 − ln(combined_with_bh)@0.73`.
- Paired ratios against arm A1, per event: `ρ_e ≡ Δ_e^{A2}/Δ_e^{A1}` and
  `τ_e ≡ t_e^{A2}/t_e^{A1}` (guard `|denominator| < 1e-6`, count and report
  guarded events, as `paired_check.py` already does).
- **Strata** (fixed partition of the 534 joint_r1 dark pairwise survivors, taken
  from the existing `gate_vii` sets — not recomputed under A2, or the partition
  would move with the treatment):
  - **shared-218** — dark survivors in both iiib and joint_r1. Reference
    quantities under A0: Σ Δ_e = **−112.6967467481951**, mean −0.5169575538908032.
  - **resurrected-316** — joint_r1-only dark survivors ("scatter-resurrected
    deep-tail"). Reference Σ Δ_e under A0 = **−492.0762245626546** = 81.37% of
    the joint headline −604.7729713108497; per-event 3.0122× steeper than
    shared-218.

### Primary reads (reported for BOTH venues, always together)

1. **Paired distribution:** median `ρ_e`, 16th/84th percentiles, Spearman
   `ρ_e`-rank correlation between arms, fraction `|ρ_e − 1| < 0.05` and `< 0.20`,
   count of guarded events. Same battery for `τ_e`.
2. **Stratum decomposition:** Σ Δ_e and mean Δ_e under A1 and A2, separately for
   shared-218 and resurrected-316, with the **relative** move
   `m_S ≡ |ΔΣ_218|/112.697` and `m_R ≡ |ΔΣ_316|/492.076`, signs reported.
3. **Context only, never a criterion:** the 2D MAP and the full-grid posterior
   mean ± sd per arm. Reported because the reader will ask; the branch does
   **not** turn on it (per-event 0.3–0.5σ rails vs class-summed +3.4–6.1σ — the
   MAP is a screen, not a gate).

### Branches

- **(a) TAIL-ACTING** — `m_R ≥ 0.25` **and** `m_S < 0.10`, **and** the paired
  `ρ_e` distribution on shared-218 is centred within ±10% of 1.0 (median, with
  the 16/84 band reported) ⇒ **D1 acts on the scatter-resurrected deep-suppression
  tail, not on the robust core.** Materially more benign than owning the 2D MAP:
  D1 becomes a *composition* systematic on a stratum already known to be
  pathological, and the follow-up is a stratum-restricted robustness quote, not a
  `/physics-change`.
- **(b) CORE-REACHING** — both strata move coherently: `m_S ≥ 0.25` **and**
  `m_R ≥ 0.25` **with the same sign**, **and** median `ρ_e` on shared-218 outside
  [0.9, 1.1] ⇒ **D1 reaches the core object.** Escalates: the selection objects
  are mis-specified in a way that touches robust events, and the remedy routes to
  `/physics-change` (selection-function definition), author-gated.
- **(c) MIXED / UNDETERMINED — first-class, non-forcing.** Anything else:
  one stratum in band and the other intermediate; opposite signs between strata;
  a venue split; `m_S` and `m_R` both < 0.10 (a **bounded null** — report the
  bound, do not promote a mechanism); or the distribution and the sums disagree.
  **Read the split directly and report it; do not force a branch.** Specifically:
  opposite-sign strata are a *finding in themselves* (that is exactly the
  cancellation A2 was written for) and must be reported as such.

**Anti-tuning:** these thresholds (**0.10 / 0.25 / ±10 %**) are fixed here,
computed mechanically from the definitions above, and **may not be adjusted after
the readout**. They are locked by this commit: the git object of this file is the
evidence of what was registered, and any later change to a threshold is visible
as a diff and is by construction an amendment, not a registration. Both floored
and unfloored likelihood readouts are reported where the zero-handling strategy
applies. The S4 ceiling (0.0107 / 0.0342 nats, ≥90 % compliance) is likewise
locked at this commit, and it is **not** one of the branch thresholds.

### Secondary pre-registered reads

**Expected NULLs** (a difference is itself a finding, and voids the run until
explained):

- **N1 — horizon invariance.** `dl_max(h=0.73)` must be **9.164987215485882** in
  arms A1 and A2, identical to A0 — already measured for all three pools
  (`cand_b_joint_selection_results.json:grids`). If the S_and pool moves the
  horizon, the derived pool was built wrong.
- **N2 — catalogue-leg likelihoods.** `L_cat_no_bh` and `L_cat_with_bh` are
  numerator objects and carry no selection function (p_det-inside is not the
  production convention). Expected **bit-identical** between A1 and A2 for all
  41 × 1588 cells. If they differ, selection has leaked into the numerator and the
  run is void.
- **N3 — A1 vs A0 composition near-null.** Dropping 3.0% of the pool (1 426
  SNR-passers) should perturb the selection scalars at the sub-percent level.
  Registered expectation: `|Δ ln w̃_G(h)| < 0.01` at every h and 2D MAP unmoved by
  more than one grid step. A large A1-vs-A0 move means the pool composition, not
  D1, dominates — and the A2-vs-A1 contrast must then be re-interpreted, not
  discarded.
- **N4 — 1D bit-identity is NOT expected.** Stated explicitly to prevent a false
  null: `β_Ḡ^φ` and `D̃^φ` feed the 1D mixture too, so the 1D channel **will**
  move under A2. (Contrast with the frozen-g run, where 1D bit-identity *was* the
  correct expectation.) A 1D channel that is bit-identical under A2 would mean
  the selection substitution did not take effect.

**Directional sub-predictions, conditional on the leading mechanism:**

- **S1 — self-certification of the treatment.** Recomputed from the A2 run's own
  objects at h = 0.73: `Σ⁴ᴰ_A2/Σ⁴ᴰ_A1` must reproduce **0.2286246597604769** and
  `β_Ḡ^φ_A2/β_Ḡ^φ_A1` must reproduce **0.3129747690740832**, each to ≤1e-3
  relative. Check this **before reading any tilt**.
- **S2 — conditional on C7 (and on B3 finding `g_frac` non-invariant):** `g_frac`
  must move under A2, and its h-slope (`Δln ḡ` across the grid, A0 reference
  0.047586, bit-identical across venues) must change. If `g_frac(h)`'s slope is
  unchanged to ≤5%, D1 does not act through the completion leg's mass factor, and
  the C7 convergence is refuted regardless of the branch.
- **S3 — sign expectation, stated so it can fail:** in-catalogue hosts are
  *heavier* than the pass-band (`s_G < s_D`), so band-aware objects should
  **increase** the in-catalogue weight relative to the dark leg. A move in the
  opposite direction is a finding and must be reported, not smoothed.
  **B2 sharpens this into a direction in h:** `ln(s_G/s_D)` is monotone
  *decreasing* in h (C10), so the in-catalogue re-weighting is **stronger at low
  h than at high h**, and the A2−A1 mixture-tilt shift must therefore be
  **one-signed across the grid**. A sign flip in `t_e^{A2} − t_e^{A1}` as a
  function of h, at fixed event, contradicts the first-order mixture-log-odds
  account and is a registered *failure* of S3.

- **S4 — the B2 mixture-log-odds CEILING (this is the absolute scale B2 was run
  to supply).** To first order the treatment enters the per-event mixture only
  through the log-odds of the two legs:
  `Δ ln(w̃_G/(1−w̃_G)) = ln(s_G/s_D)(h)` — the class-conditional retention is a
  **common multiplicative rescaling** of `Σ⁴ᴰ` and `β_Ḡ^φ` at each h, and it
  cancels out of any event that is purely one class. It follows mechanically that
  the induced shift in a per-event **h-difference** tilt is bounded by the
  corresponding change in that log-odds:
  - for the gate-(vii) pair (0.73 → 0.81):
    `|t_e^{A2} − t_e^{A1}| ≤ |ln(s_G/s_D)(0.81) − ln(s_G/s_D)(0.73)|`
    = **0.01066732831245143 nats**;
  - over the full grid (0.60 → 0.86): **0.03418233182790781 nats**.

  **Registered band S4:** at least **90 %** of un-guarded events must satisfy the
  0.73→0.81 ceiling, `|t_e^{A2} − t_e^{A1}| ≤ 0.0107 nats`, in **both** venues.
  A larger violating fraction means the substitution reaches the posterior through
  a channel that is *not* the mixture log-odds — the leading candidate being the
  completion leg's mass factor `g_frac`/`B_num_wbh` (C7, and S2) — and the readout
  must then say so explicitly rather than attributing the move to the mixture
  weight. **S4 is diagnostic-of-channel, not a branch criterion:** it does not
  gate (a)/(b)/(c), which remain purely relative.

  Corollary registered as a **scale sanity check, not a criterion:** if the shift
  were fully coherent across all 1588 events at the ceiling, the class-summed
  0.73→0.81 move would be ≤ 1588 × 0.0107 ≈ **17 nats**. A measured class-summed
  |ΔΣ| far in excess of that (≳ 2×) cannot be a mixture-log-odds effect at all and
  must be reported as an unexplained channel. Recorded because amendment A2
  forbids reading Σ alone — this is a ceiling on Σ, never a branch trigger.

**Provenance/era guards:**

- All diagnostics CSVs must be single-sweep: `n_rows == 41 × n_events` per run
  (the post-fix twins are 65 108 = 41 × 1588). Any 2× row count is the
  concatenated-era trap — disambiguate or discard.
- No second differences across the h-grid seams (non-uniform grid).
- `run_metadata.json` must record the pool path actually used; verify it before
  reading, as the frozen-g run verified its flag.

### Band status at registration

**CLOSED by B2 — the absolute scale of an expected tilt shift.** Registered as
**S4** above: ceiling `|t_e^{A2} − t_e^{A1}| ≤ 0.0107 nats` per event on the
0.73→0.81 pair (0.0342 over the full grid), with a ≥90 %-of-events compliance
band and a ≈17-nat class-summed sanity ceiling. Derived mechanically from
`Δ ln(s_G/s_D)` (C10); no number in it is chosen.

**STILL DELIBERATELY OPEN (no invented numbers):**

- **A σ(H₀) forecast.** Not registered: the F5 engine is **N/A** for a
  selection-induced effect (unbiased by construction; mass-blind mock `p_det`) and
  no Fisher-forecast asset exists in the repo. Documented gap, not a silent
  omission.
- **A band on the 2D MAP move.** Deliberately absent — the MAP is context, not a
  criterion (see Primary read 3).

---

## Scope guard

- **No re-simulation.** CRB set and catalogues are consumed through existing
  symlinks; no waveform, Fisher matrix, or injection is regenerated.
- **The D1 constraint of record is honoured**: the catalogue stays band-passed and
  is scored only against *band-aware* objects. The `ParameterSpace.p0` retirement
  is a separate, simulation-side `/physics-change` and is not authorised here.
- **No production posterior.** All three arms are counterfactual diagnostics.
- **Any actual fix routes through `/physics-change`** — 5-item package → author
  approval → ledger rows. This file authorises a **measurement**, never a formula
  change.
- Model/effort policy for the readout: haiku/low for the mechanical extraction and
  fingerprints; opus/high for the interpretation and any adversarial pass; the
  branch call is presented to the author, never self-adjudicated.

---

Verdict to be appended below by the session that reads out the run — after this
file is committed, no edits above this line.

---

## VERDICT — appended 2026-08-05, readout session

**(c) MIXED / UNDETERMINED — bounded null, with one falsified prereg assumption
(N2) recorded as a discovered fact.**

- **Branch scoring**: `m_S = 0.0322`, `m_R = 0.0107` (joint_r1; both < 0.10 ⇒ the
  **(c) bounded-null** clause fires; the branch (a)/(b) thresholds of 0.25 are
  missed by ~8–20×). Shared-218 paired `ρ_e` median **0.983** (iiib) / **0.964**
  (joint) — within ±10% of 1. `ΔΣ(A2−A1)`: shared-218 +4.68/+3.63 nats;
  resurrected-316 +5.27 nats (joint).
- **S2 decisive**: `g_frac` is **bit-identical A1-vs-A2** per event, per h, in
  both venues (0/65108 mismatches) ⇒ the D1→`g_frac` convergence route (C7) is
  **DEAD at machine precision**.
- **S-bands**: S1 self-certification `Σ⁴ᴰ` **PASS iiib** (relative deviation
  6.8e-5) / **FAIL joint_r1** (12.5%) — `Σ⁴ᴰ` folds the catalogue-side `f_k`,
  which is legitimately venue-dependent; flagged, not adjudicated. S1
  `β_Ḡ^φ` **PASS both venues** (≤4e-6). S3 sign expectation: one-signed
  96.6% (iiib) / 94.9% (joint). S4 mixture-log-odds ceiling: **PASS**
  98.5% (iiib) / 97.5% (joint) of events within the 0.0107-nat band.
- **N2 run-voiding null FAILED, root cause traced**: `L_cat_no_bh`/`L_cat_with_bh`
  differ A1-vs-A2 in 62–69% (iiib) / 18–38% (joint) of cells — because under
  `volume_deconv`/`absolute_marginal` the catalogue-leg host-z prior carries the
  completeness callable `f_k`, and `f_k` is **built from the injection pool**
  (the very object D1 substitutes). The prereg's assumption "`L_cat` carries no
  selection function" is **falsified** by the current estimator config. Per the
  prereg's own STOP semantics, the N2-dependent reads are void as registered;
  the stratum read above is reported on the valid (shrunk) `Δ_e` populations
  (294/1588 iiib, 607/1588 joint) and is **CONDITIONAL on that caveat**. The
  `f_k`-is-pool-fed fact is promoted to the next session's intake queue.
- **N1 PASS** (horizon `9.164987215485882` exact, 41/41 h, both pools).
  **N3 PASS** (`|Δln w̃_G| ≤ 0.001`). **N4 PASS** (expected non-nulls moved).
- **Context (not scored)**: `w̃_G(0.73)` A0→A1→A2 = 0.06197→0.06191→0.05101
  (iiib) / 0.07080→0.07074→0.06542 (joint). No MAPs by design (no combine).
- **Ambiguities recorded**: S2's registered A0 reference slope `0.047586` did
  not reproduce under mean/median/sum aggregations (closest 0.040284,
  bit-identical across venues); `m_S` was applied using the joint-only
  registered denominator.
- Jobs 6152697–6152704, 492/492 COMPLETED, code `128f318a`. Evidence:
  `results/run_20260805_d1/readout.{py,json}`.

---

## AUTHOR RULING (2026-08-05)

**D1 disposition — ACCEPTED** (Jasper Seehofer, 2026-08-05, morning author
queue: `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_8.md` §1 item 4).

The author accepts the **bounded-null (tilt route) verdict** recorded above:
D1 does not reach the core 2D-bias object via the tilt route (`m_S = 0.032`,
`m_R = 0.011`, both ≪ 0.25; `g_frac` bit-identical under S_and, so the
D1→`g_frac` (C7) convergence route is dead).

Unchanged by this ruling:
- The simulation-side `ParameterSpace.p0` bounds retirement remains **its own
  future `/physics-change`** (as this file's Scope guard already states); it is
  not authorised by this disposition.
- The 3135-event catalogue is **still never re-scored band-blind** (standing
  constraint of record).

This block is append-only; no text above it was modified.
