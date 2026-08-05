# CLAIM — D1: the p0-window mass band-pass as a candidate 2D-bias owner

**Status: CLAIM, NOT ESTABLISHED. Written to be attacked.**

Opened 2026-08-05 as the **first full run of the standing Research Cycle**
(`.claude/skills/research-cycle/SKILL.md`, `docs/RESEARCH_CYCLE.md`), stage 0.
Parent: `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_7.md` §0 suspect 1 /
§1 item 2. Code base at intake: `main@a7e0d559`; code base at commit of this file:
`main@9a715405`. The intervening commits (`a7e0d559..9a715405`) are: the g_frac
derivation package + its Gate-B adjudication, the closed-loop 2-channel
calibration harness (`validation/closed_loop_gfrac.py`, new file) and its
registered run, the book ch12 page, a CHANGELOG entry, and the N-2 sel-1d
pre-registration + its `--selection_in_completion_numerator` instrumentation
toggle. The only one touching a `/physics-change` trigger file is the last, and
it is **opt-in with `default="off"` documented byte-identical to the pre-flag
production path** (`arguments.py:762-784`, `:283-296`, read this session) — so
C1–C9, all of which are statements about the production path, are unaffected.
[LOCAL]

**Status of this file: stages 0–1, COMMITTED 2026-08-05.** Stage 2 lives in
`PREREGISTRATION_D1_SAND_REWEIGHT.md` (same directory). Stages 0 and the
non-B2 parts of stage 1 rest on free re-reads of on-disk artifacts only
(amendment A1); the single new measurement folded in at commit time is **B2**
(claim C10 below), the measurement-before-gate read that stage 1 named and
stage 2's STEP 0 required to run first. Its instrument and output are committed
alongside: `d1_b2_sand_hslope.py` / `d1_b2_sand_hslope.json`.

**Append-only from here.** Nothing above the stage-3 verdict line may be edited
after this commit; corrections are appended, never overwritten.

---

## CONSTRAINT OF RECORD — binding, quoted verbatim from RUNBOOK-7 §1.2b

> the existing 3135-event catalogue stays band-passed and must never be
> re-scored against band-blind objects; the p0-bounds retirement is
> simulation-side, for future campaigns only.

Consequences that bind every downstream stage of this investigation:

1. **The permitted direction is band-AWARE, never band-blind.** Making the
   inference's selection objects model the filter the pipeline actually applied
   (`S_and = P(SNR ≥ 20 ∧ p0 ∈ W | d_L, M_z)`) is *consistent* re-scoring and is
   the allowed counterfactual. Re-scoring the band-passed catalogue against
   objects built from `SNR ≥ 20` alone — i.e. removing the band-pass from one
   side only — is forbidden, because that is precisely the inconsistency under
   investigation, and a posterior produced that way would be quotable by
   accident.
2. **The `ParameterSpace.p0` bounds retirement is a separate, simulation-side
   `/physics-change`** (RUNBOOK-7 §1.2b). It is *not* part of this investigation's
   measurement; it changes future campaigns, never the catalogue of record.
3. **No production posterior may be emitted by this thread.** Everything stage 3
   produces is a diagnostic counterfactual against its unfrozen twin, on the
   pattern of `results/run_20260804_postfix/gate_vii/PREREGISTRATION_FROZEN_GFRAC.md`
   ("No production posterior is produced").

---

## Two-layer exoneration check (standing rule, run BEFORE opening)

Performed 2026-08-05, both layers, verbatim reads:

| layer | file | read |
|---|---|---|
| 1 (local) | `results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md` `## Exonerated — do NOT re-open without new evidence` (:191-204 + the 07-30 appendices) | full section |
| 2 (project-wide, authoritative) | `.../gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 "DO NOT RE-TRY", incl. all 17 ⚠ items and the standing scoping rule | full section |

**Result: NOTHING ISOMORPHIC TO D1 APPEARS ON EITHER LAYER.** [LOCAL]

D1 was discovered 2026-08-04 (`fixb_x15_attribution/CAND_B_CRB_FILTER.md`), after
both layers were compiled (2026-07-30), so this is the expected outcome; it was
checked, not assumed. The ⚠ items — the live re-litigation risk — were read
individually. The four **nearest neighbours**, and why each is a different
object:

| nearest exonerated item | layer | why D1 is not it |
|---|---|---|
| "p_det estimator choice" and "p_det inside/outside" | 1 | Both are questions about *which* estimator/placement is used for a selection function the inference *has*. D1 is a filter the pipeline **applies and no selection object represents at all**. `FIXB_PATHA_PACKAGE.md` (banner) records the same distinction against the same binding list (`HANDOFF_20260730.md` §3). |
| **HB** — "hard mass window as support truncation" (tilt −0.317 nats, sign-inverted, 40–50× too small) | 1 | HB is the **estimator-side** candidate-host mass window in `handler.py` (`:605`), applied inside the likelihood. D1 is a **generator/CRB-stage** window in an *orbital* parameter (`p0`) that acts on `M_z` only indirectly, and it acts on the **event catalogue**, not on the candidate list. Different file, different stage, different variable. |
| "candidate-window **membership**" (exact removal moves MAP 0.81→0.82, wrong sign) | 1 | Same estimator-side object as HB, tested by removal. Says nothing about a filter upstream of the CSV the estimator reads. |
| ⚠4 "Hard support truncation / hard clamp **in production**" (#63); ⚠1 `mass_trunc` kernel as the 2D driver | 2 | Both are *kernel/likelihood-support* hypotheses in the inference. D1 is a **selection-function inconsistency between generator and estimator** — the class the ledger's own §2 does not contain, and the class RUNBOOK-7 §2 explicitly says SBC cannot see. |

**Scoping caveat carried forward** (ledger §2 standing rule): `volume_trunc`
(#70) and `mass_trunc` (#72) were both exonerated on the **same seed-600
494-event shallow subsample**. Neither may be cited against D1 as a universal
negative, and D1's venue (campaign-51, 3135 events, 1588 evaluated) is not that
venue.

---

## Claims

Tag vocabulary per `docs/RESEARCH_CYCLE.md` stage 0: `[LOCAL]` re-measured or
re-derived in this session · `[DOC]` read from a committed artifact ·
`[INFER]` inference from `[LOCAL]`/`[DOC]` · `[AGENT]` measured by a subagent and
**not** independently reproduced. **No claim below is `[AGENT]`** — nothing in
this file rests on an un-reproduced subagent number. Where a number originates in
a 2026-08-04 measurement JSON, the JSON is `[DOC]` and any arithmetic on it done
here is `[LOCAL]`.

### C1 — The post-SNR filter of campaign 51 is a hard `p0 ∈ [10.002, 15.998]` window, not a CRB/precision filter [LOCAL + DOC]

`ParameterSpace.p0` carries `lower_limit=10.0`, `upper_limit=16.0`,
`derivative_epsilon=1e-3` and a `[PHYSICS]` comment stating the bounds are
"SNAPSHOT-mode bounds only … RETIRED as the production convention on 2026-07-28"
while production draws `p0` from the plunge window with **no upper clamp**
(`master_thesis_code/datamodels/parameter_space.py:95-113`, read this session)
[LOCAL]. `five_point_stencil_derivative` raises `ParameterOutOfBoundsError`
whenever `value ± 2ε` leaves the declared bounds
(`master_thesis_code/parameter_estimation/parameter_estimation.py:268-276`, read
this session) [LOCAL], and `main.py:772-779` turns that into a post-SNR event
drop [DOC]. Effective survival window: `p0 ∈ [10+2e-3, 16−2e-3]`.
Per-event trace of the runs of record: **8 345 / 12 039 = 69.3 %** of SNR-passers
rejected at this guard, `p0` the failing parameter in 8 345 / 8 355 = 99.88 % of
out-of-bounds rejections; all other post-SNR loss 154 / 12 039 = 1.28 %
(`CAND_B_CRB_FILTER.md` §0–§2) [DOC]. All 3 135 saved rows satisfy
`p0 ∈ [10.0025, 15.9939]`, flush against both edges [DOC].

**Refute by:** re-run the per-event `simulate_*.err` trace parser
(`remote_trace_parse.py`) on the archived task logs and show the failing-parameter
histogram is not ≥99% `p0`, or show any saved row with `p0` outside the window.
Cheapest form: `awk` the `p0` column of `prepared_cramer_rao_bounds.csv` for
min/max — one command, decisive on the second half.

### C2 — The window is a band-pass in detector-frame mass `M_z`, centred near `M_z ≈ 3e5–1e6` [DOC + LOCAL]

`p0` is a steep **decreasing** function of `M_z` at fixed `t_plunge`
(`p0(t_plunge = 4.5 yr)` = 109.5 / 34.3 / 10.83 / 6.81 / 4.71 at
`M_z` = 1e4 / 1e5 / 1e6 / 3e6 / 1e7,
`docs/derivations/plunge_window_initial_conditions.md`) [DOC], so a fixed `p0`
window selects a mass band. Measured on the gate-free injection pool:
`q = P(p0 ∈ W | M_z, SNR ≥ 20)` = 0.002 (1e4–3e4) → 0.13 (2e5–3e5) →
**0.64–0.77 (5e5–1e6)** → 0.40 (1e6–2e6) → 0.017 (2e6–3e6) → 0.0004 (3e6–1e7) →
0 (>1e7) [DOC]. The catalogue-side transfer is in
`cand_b_joint_selection_results.json:incat_leg.by_Mz.q`, re-read this session:
0.0145 / 0.0669 / 0.202 / 0.681 / 0.709 / 0.743 / 0.338 / 0.0268 / 3.6e-4 /
3.8e-10 over the `M_z` edges [0, 1e5, 2e5, 3e5, 5e5, 7e5, 1e6, 2e6, 3e6, 1e7, ∞]
[LOCAL re-read of a `[DOC]` artifact].

**Refute by:** show `q(M_z)` flat to ±10% across the pool's occupied `M_z` range
(i.e. the window is effectively class-blind). One pass over the 707-file pool,
no estimator involved.

### C3 — No inference selection object models the window [DOC + LOCAL]

Every selection object the inference builds — `Σ³ᴰ`, `Σ⁴ᴰ`, `β_G`, `β_Ḡ`, `D`,
`p_det` — is constructed from the injection pool's `SNR ≥ 20` alone
(`CAND_B_CRB_FILTER.md` §1, §3 mismatch #1) [DOC]. `SimulationDetectionProbability`
takes an injection directory and thresholds on SNR; there is no `p0` argument
anywhere in its constructor signature, which is why the D1 instrument has to
*fake* the joint selection by setting `SNR := 0` on p0-rejected injections
(`cand_b_joint_selection.py:90-94`, read this session) [LOCAL].

**Refute by:** exhibit any code path in `simulation_detection_probability.py` or
`bayesian_statistics.py` that reads or filters on `p0`. `grep -rn "p0"
master_thesis_code/bayesian_inference/` — one command.

### C4 — The band-pass is class-conditional: `s_G / s_D = 0.7305` [LOCAL arithmetic on a DOC artifact]

Measured with the production estimator machinery on the injection pool of record
(707 files, 200 100 rows; grids `dl_max = 9.164987215485882` identical for the
S, S_and and reference pools — i.e. the horizon anchor does not move):
in-catalogue leg `s_G = Σ⁴ᴰ_and/Σ⁴ᴰ = 85 742 970.216 / 375 038 153.392 =
0.2286246597604769`; dark leg `s_D = β_Ḡ^{4D,φ}_and/β_Ḡ^{4D,φ} =
278 032 484.941 / 888 354 309.723 = 0.3129747690740832`
(`cand_b_joint_selection_results.json`) [DOC]. Ratio recomputed this session:
**0.2286246597604769 / 0.3129747690740832 = 0.7304891075943567** [LOCAL].
In-catalogue hosts are *heavier* than the pass-band and are therefore
preferentially rejected — the sign is **favourable** to the in-cat over-prediction
that Fix B's gate (ii) was failing on.
Its reproduction anchors: `β_Ḡ^{4D,φ}` reproduces the path-A record to
−5.6e-5, `Σ⁴ᴰ` to −1.2e-3 [DOC].

**Refute by:** rebuild the two derived pools with a different but still
production-legitimate `p_det` estimator/binning (the ratio is documented as
estimator-dependent: grid 0.730 vs refuted raw-horizon 0.919,
`FIXB_PATHA_PACKAGE.md` §1.3 rule 4) and show the *grid-convention* value moves
outside ~0.73 ± 0.03. Note the arbitration that fixed the grid convention (KS
D = 0.114/0.047 vs 0.49/0.50 against the detections' own `M_z` distribution) must
be re-run by anyone who changes the estimator.

### C5 — The band-pass is the causal bulk of the in-catalogue absolute-count excess [DOC]

Candidate (b) owns **×1.342** of the ×1.50–1.69 in-cat over-prediction — 56.4% of
`ln(×1.6856)`, 72.9% of `ln(×1.4992)` — leaving a residual **×1.068 ± 0.083 (stat)
± 0.047 (sys) = 0.7σ from unity**; the competing candidates are (a) REFUTED
(×1.009, wrong sign) and (c) mechanism-confirmed/remedy-refuted (×1.099)
(`FIXB_PATHA_PACKAGE.md` §0–§1.2, verified there by the adjudicator at 27/27
checks ≤2e-4) [DOC]. Under S_and scoring the class-share discriminator moves from
z = −4.42σ (SNR-only) to **z = −0.48σ** (joint), i.e. gate (ii)'s −3.7σ closes
*only* by conditioning on this filter [DOC]. Gate (ii-d), the absolute
detected-count audit, is closed by exactly this attribution [DOC].

**Refute by:** the residual is a bound, not a proof (it carries +4.35% forward-model
closure error and a single catalogue realization). Show the closure test
`s_mixture_pred = 0.3068` vs realized `3540/12039 = 0.2940` disagrees by ≫5% under
a second realization, and C5's size collapses into "unattributed" again.

### C6 — A mass band-pass at fixed source mass is a redshift-selection distortion, i.e. the *shape* required to own a 2D-channel residual [INFER]

`M_z = M(1+z)`. A filter that is sharply non-flat in `M_z` therefore selects in
`z` at fixed `M` — with a `z`-dependence that the 1D channel never sees (the 1D
numerator carries no mass density: `cov_obs = cov_4d[:3,:3]`,
`bayesian_statistics.py:2495`, gate (iv) PROVEN) [DOC]. Hence D1 can distort the
2D channel and leave the 1D channel's *catalogue leg* structurally untouched —
the observed asymmetry (2D residual +0.05–0.07 at MAP at the idealized venue;
1D railed at 0.600) is *consistent* with, though not evidence for, D1.
This is an inference about **shape compatibility only**. It is not a measurement,
and the sign and size of D1's effect on the 2D MAP are **unknown**
(see "What is explicitly NOT claimed").

**Refute by:** the stage-2 pre-registration's S_and re-weight — if per-event 2D
tilts are unmoved (per-event `r_e` distribution centred on 1.0 within its 16/84
band) in **both** strata, C6's compatibility is empty and D1 is not a 2D-bias owner
at this venue.

### C7 — There is a live convergence route: the same selection machinery feeds the object now known to carry the residual 2D displacement [DOC + INFER]

The frozen-`g_frac` pre-registration returned **CONFIRM in both venues**: freezing
each event's `g_frac = B_num_wbh/B_num` at its h = 0.73 value moves the 2D MAP
0.780 → **0.660** (iiib) and 0.800 → **0.640** (joint r1), live-equals-CSV-proxy to
0 grid steps, with 1D bit-identity and `w̃_G`/`r_Malm`/`α_G^φ`/`D̃^φ` bit-identical
(`results/run_20260804_postfix/gate_vii/PREREGISTRATION_FROZEN_GFRAC.md`, VERDICT
appended 2026-08-05) [DOC]. `g_frac` is the completion leg's population-mass-density
factor; `B_num_wbh` is built from the **with-BH-mass detection object**, which is
exactly the object D1 says is built on the wrong selection (`S`, not `S_and`)
[INFER]. That verdict names the convergence itself ("possibly with D1 — same
selection machinery feeds `B_num_wbh`") [DOC].

**Refute by:** show that `B_num_wbh`'s mass factor does not depend on the
with-BH-mass survival object — i.e. that replacing `S_4D` by `S_and,4D` leaves
`g_frac(h)` unchanged to ≤1e-6 per event. This is a *cheap analytic/one-pass*
check and, per measurement-before-gate, should run before any expensive re-score:
if `g_frac` is invariant, C7 dies and D1 cannot reach the 2D MAP through this route.

### C8 — The venue's joint 2D-vs-1D headline is composition-dominated by a deep-tail stratum that D1 selects on [LOCAL]

Recomputed this session from
`results/run_20260804_postfix/gate_vii/{paired_check.json, gate_vii_readout.json}`
(free re-read, amendment A1; arithmetic done here, hence [LOCAL]):

| quantity | value |
|---|---|
| dark pairwise survivors, iiib / joint_r1 | 219 / 534 |
| shared (paired) stratum | **218** |
| "resurrected" stratum = joint-only | 534 − 218 = **316** |
| joint headline `Σ_dark Δln(L²ᴰ/L¹ᴰ)`, 0.73→0.81 | −604.7729713108497 |
| shared-218 contribution in joint (218 × mean −0.5169575538908032) | **−112.6967467481951** |
| resurrected-316 contribution | −492.0762245626546 = **81.37 %** of the headline |
| per-event steepness, resurrected vs shared | **3.0122×** |
| shared-stratum dilution, joint vs iiib (−0.51696 / −1.10160) | **0.4693×** |
| paired ratio `r_e` on the shared 218 | median 2.0825, 16/84th 1.3137 / 3.2972, Spearman 0.5281 (p 4.7e-17), only 1.4% within ±5% |

The 316 resurrected events are events whose 2D catalogue leg was *dead* at the
idealized venue and became alive under scatter — i.e. deep-tail, faint,
low-catalogue-support events. D1 selects on `M_z` and therefore is a live candidate
for shaping exactly which events populate that tail [INFER on the last sentence
only]. **This is why amendment A2 forbids scoring D1 on Σ alone.**

**Refute by:** show the resurrected-316 stratum has the same `M_z` distribution as
the shared-218 stratum (KS on `M_z` from `prepared_cramer_rao_bounds.csv`,
zero-compute) — then D1's mass selection is orthogonal to the stratum split and
C8's relevance to D1 collapses (the composition finding itself survives; only its
D1-relevance dies).

### C9 — Nothing about D1 has been measured on the 2D posterior [LOCAL, negative claim]

No artifact in the tree scores the 2D channel under S_and-consistent selection
objects. `cand_b_joint_selection.py` computes selection **scalars** (`s_G`, `s_D`,
`Σ⁴ᴰ_and`, `β_Ḡ^φ_and`, closure shares) at **h = 0.73 only** — it does not touch
the estimator's per-event likelihoods and does not produce a posterior
(verified by reading the script end-to-end this session) [LOCAL]. Therefore every
statement about D1's effect **on H₀** is currently unmeasured.

**Refute by:** exhibit any committed artifact containing a 2D MAP computed with
band-aware selection objects. (If one exists, stage 2 is redundant.)

### C10 — The class-conditional retention `s_G/s_D` is **h-DEPENDENT**: the h-flat null is falsified, and D1's posterior-tilt route stays live [LOCAL]

**This is the stage-1 measurement-before-gate read (B2), completed 2026-08-05.**
Instrument: `d1_b2_sand_hslope.py` (this directory) — the selection-scalar
computation of `cand_b_joint_selection.py` re-run **unmodified in every other
respect** (same two derived pools, same production
`SimulationDetectionProbability`, same `FLAGS`, same in-catalogue GLADE+ mass-density
sum and same φ-quadrature dark leg) but **swept over h** instead of pinned at
h = 0.73. Output: `d1_b2_sand_hslope.json`. Wall time 274.86 s.

**Pin reproduced exactly.** At h = 0.73 the sweep returns
`s_G = 0.2286246597604769`, `s_D = 0.3129747690740832`,
`s_G/s_D = 0.7304891075943567`, `dl_max = 9.164987215485882` — **relative
deviation 0.0 from C4's pinned values on all four** (`pin_reproduced: true`).
The instrument is therefore certified against the artifact it generalises.

Measured over the canonical 41-point h grid:

| h | `s_G` | `s_D` | `s_G/s_D` | `ln(s_G/s_D)` |
|---|---|---|---|---|
| 0.60 | 0.23965338570393016 | 0.32258528423556976 | **0.7429148117274993** | −0.29717389530944144 |
| 0.73 (pin) | 0.2286246597604769 | 0.3129747690740832 | **0.7304891075943567** | −0.31404095879323346 |
| 0.81 | — | — | 0.7227381549328574 | −0.3247082871056849 |
| 0.86 | 0.21936221189053678 | 0.3055399460246244 | **0.7179493704330814** | −0.33135622713734925 |

- **Monotone decreasing across all 41 grid points** (verified pointwise this
  session; the finite-difference slope stays in
  [−0.13355779435944015, −0.12410918801212245] with no sign change).
- `Δ ln(s_G/s_D)` over the full grid = **−0.03418233182790781**
  (`grid41_delta_ln_ratio_max_minus_min` = 0.03418233182790781, endpoint form
  identical — a consequence of the monotonicity).
- Endpoint slope `d ln(s_G/s_D)/dh` = **−0.13147050703041463** per unit h.
- **Horizon invariance holds at every h:** `dl_max_S = dl_max_and =
  9.164987215485882` for all 41 points — the S_and pool does not move the horizon
  anchor, so the tilt is a genuine class-conditional effect and not a grid artefact.
- Over the gate-(vii) comparison pair used by stage 2's `Δ_e`/`t_e` definitions,
  `ln(s_G/s_D)(0.81) − ln(s_G/s_D)(0.73) = −0.01066732831245143`.

**Verdict against stage 2's registered STOP clause.** STEP 0 registered: stop and
downgrade to a bounded null if `|Δ ln(s_G/s_D)|` across the grid is **< 0.005**
(0.5 %). Measured **0.0342 = 6.84× the band**
(`h_flat_verdict_full_grid: false`, `h_flat_verdict_endpoints: false`).
**The h-flat null is FALSIFIED; the STOP clause does not fire; the three-arm
re-weight is authorised to run.** [LOCAL]

Direction: `s_G/s_D` *decreases* with h, i.e. the band-pass removes in-catalogue
mass-density **increasingly** relative to the dark leg as h rises. Since
`Δ ln(w̃_G/(1−w̃_G)) = ln(s_G/s_D)` to first order in the mixture log-odds, a
band-aware estimator shifts weight toward the in-catalogue leg **more at low h
than at high h** — a coherent, monotone, one-signed tilt across the whole grid.
That is the *shape* of a posterior-displacing systematic. Its **size on H₀ remains
unmeasured** (C9 stands); B2 bounds the mechanism's log-odds lever, not its MAP.

**Refute by:** (i) show the sweep's h-dependence is an artefact of the derived-pool
construction rather than of the selection — rebuild the two pools with a different
but production-legitimate `p_det` binning/estimator (C4's `Refute by:` convention)
and show `Δ ln(s_G/s_D)` over the grid falls below 0.005; or (ii) exhibit an error
in the sweep by re-running `d1_b2_sand_hslope.py` and failing to reproduce
`pin_reproduced: true`. Either kills C10 and reinstates the bounded-null route.
Cheapest form: re-run the script (274 s, no cluster).

1. **Not claimed: that D1 biases H₀, in any direction, by any amount.** C6 claims
   *shape compatibility*; C5 claims an *absolute-count* attribution. Neither is a
   posterior displacement. The sign is genuinely open: the count effect is
   *favourable* (C4/C5), which is not the same as favourable for the MAP.
2. **Not claimed: that D1 owns the residual 2D displacement.** The frozen-g run
   (C7) shows `g_frac(h)` *carries* it; whether D1 *causes* `g_frac`'s h-slope to
   be wrong is precisely what is unmeasured.
3. **Not claimed: that the p0-bounds retirement fixes anything about the existing
   catalogue.** It cannot: it is simulation-side, for future campaigns only
   (constraint of record).
4. **Not claimed: that the 3135-event catalogue is invalid.** It is a
   self-consistent sample of `SNR ≥ 20 ∧ p0 ∈ W`; the defect is that the
   *inference* does not know that.
5. **Not claimed: that C5's residual ×1.068 proves closure.** It is a bound
   carrying +4.35% forward-model error on one realization.
6. **Not claimed: any exoneration.** Nothing here retires any suspect; the in-cat
   class tension (RUNBOOK-7 §0 suspect 2) is untouched.
7. **Not claimed: that gate (ii) is restored as physics evidence.** It remains a
   monitored consistency number, conditional on the generator-closure convention
   **and** on modeling the very filter under investigation.

---

## STAGE 1 — Information forecast: applicability of the F5 σ_z/σ_M engine

**Verdict: N/A — documented honest gap, with a named cheap substitute.**

The stage-1 asset (`docs/SIGMA_Z_SIGMA_M_FORECAST.md`,
`scripts/bridge_closure/sigma_z_sigma_M_forecast.py`) is a **self-consistent
closure that is unbiased by construction**; its own §1 and §7 say so explicitly:
"Because the closure is unbiased, the posterior RMSE-to-truth is a clean measure
of information content" and "it measures *information content*, not production
bias mechanisms (Malmquist, MC-denominator noise, photo-z systematics)"
(`docs/SIGMA_Z_SIGMA_M_FORECAST.md:21-26`, `:145-146`) [DOC]. Two structural
reasons it cannot bound D1:

1. **Unbiased by construction** — the inference kernels use the *same* σ_z, σ_M as
   the injection, so no generator/estimator selection mismatch exists in the
   engine's universe. D1 *is* a generator/estimator selection mismatch. The engine
   would return the same σ_eff with and without a D1-like filter, by construction.
2. **Its mock `p_det` is mass-blind** — §7 caveat (a): "the selection denominator
   is **mass-blind** (shared with 1-D), not production's 4-D mass-dependent MC
   `p_det`" (`:149-150`) [DOC]. A mass band-pass is unrepresentable in an engine
   whose selection function has no mass argument.

Also on the record: RUNBOOK-7 §2 names "Fisher forecasts" for stage 1; **no
Fisher-forecast asset exists in this repo** (`docs/RESEARCH_CYCLE.md` stage 1,
"GAP — TO BUILD"). Building one is not warranted for D1: a Fisher forecast bounds
*precision*, and D1 is a *selection* question.

**What WOULD bound the effect cheaply instead — the S_and instrument already
built.** `cand_b_joint_selection.py` already measures the class-conditional
retention `s_G/s_D = 0.7305` (C4) and the derived selection scalars at h = 0.73.
Three cheap bounds are available *before* any posterior is recomputed:

- **B1 (zero new compute).** The h-slope of the mixture weight is the lever of
  record; `w̃_G` and `r_Malm` are already emitted per-h in both post-fix
  `event_likelihoods.csv`. Rescaling `Σ⁴ᴰ → Σ⁴ᴰ_and` and `β_Ḡ^φ → β_Ḡ^φ_and` with
  the measured `s_G`, `s_D` at h = 0.73 gives a **first-order** shift in `w̃_G`
  and hence a nats-per-unit-h budget, convertible to Δh via the project's own
  conversion (Δh ≈ Δnats · σ_h²/Δh_window ≈ 4.9e-3 per nat over a 0.08 window;
  see the claim file's "Errors made this session" item 1, whose *corrected* form
  this is).
- **B2 (one pool pass, ~grid-build cost). — RUN 2026-08-05; see C10.** Re-run
  `cand_b_joint_selection.py`'s selection legs across the h grid instead of at the
  single pinned h. The quantity that matters is not `s_G/s_D` itself but its
  **h-slope**: a filter with an h-independent retention ratio cannot tilt a
  posterior. `d ln(s_G/s_D)/dh ≈ 0` would have bounded D1's posterior effect at ≈0
  without ever re-scoring an event. **This was the measurement-before-gate
  candidate and it ran first, before stage 3.**
  **Outcome: NOT flat.** `d ln(s_G/s_D)/dh = −0.1315` per unit h, monotone over all
  41 grid points, `Δln = −0.0342` end to end — **6.84× the registered 0.005
  h-flat band**. The bounded-null route is closed; the tilt route is live.
  (Instrument `d1_b2_sand_hslope.py`, output `d1_b2_sand_hslope.json`, pin
  reproduced to relative deviation 0.0.)
- **B3 (zero new compute).** The `g_frac`-invariance analytic check of C7's
  `Refute by:` — if `g_frac` is invariant under `S → S_and`, the C7 route is dead.

The proper stage-1 "expected value" for this investigation is therefore
**not a σ(H₀)** but a **pre-registered band on `d ln(s_G/s_D)/dh`**, which stage 2
adopts (see the pre-registration's secondary reads). The band itself was fixed
*before* B2 ran — the h-flat threshold `|Δ ln(s_G/s_D)|` over the grid `< 0.005` —
and B2 then measured 0.0342 against it (C10). Inventing a numeric band for an
unmeasured slope after the fact would have violated the anti-tuning discipline;
the ordering here (band registered → measured → 6.84× outside) is the intended one.

**Stage-1 verdict, recorded:** the information forecast for D1 is *not* a σ(H₀)
(engine N/A, no Fisher asset) but the measured log-odds lever
`Δ ln(s_G/s_D) = −0.0342` over the production h grid, monotone and one-signed.
This is the scale that stage 2's registered bands are expressed against.

---

## Errors to avoid in this thread — do not inherit them

1. **Do not score D1 on Σ alone.** Amendment A2 exists because a Σ-agreement of
   2.6% across venues was a pure coincidence of two opposing strata (C8). Every
   comparison in stage 3 must carry a paired/stratified per-event read.
2. **Do not conflate "favourable for the count" with "favourable for the MAP".**
   C4/C5's `s_G/s_D < 1` closes a *rate* discrepancy; its posterior sign is
   unmeasured.
3. **Do not quote `s_G/s_D` without its estimator convention.** It is
   p_det-estimator-dependent (0.730 grid vs 0.919 raw-horizon, the latter refuted)
   — `FIXB_PATHA_PACKAGE.md` §1.3 rule 4.
4. **Do not use the root/idealized diagnostics CSVs without era disambiguation** —
   the pre-post-fix concatenation trap (`CLAIM_2D_BIAS_20260730.md`, new
   observation (ii)). The post-fix CSVs used for C8 are single-sweep
   (65 108 = 41 × 1588, verified in `gate_vii_readout.json:fingerprint`).
5. **Do not take second differences across the h-grid seams** — the grid is
   non-uniform (0.01 / 0.005 / 0.01).
6. **60 of the 707 pool files carry no `p0` column** (pre-plunge-window
   `code_rev a9f29e82`, 6000 rows, 3.0% of the pool, 1426 of them SNR ≥ 20).
   `cand_b_joint_selection.py` drops them from **both** derived pools so numerator
   and denominator see an identical injection population
   (`cand_b_joint_selection.py:79-88`). Any re-implementation must preserve that
   symmetry or it manufactures a spurious retention ratio.

---

## Exonerated — do NOT re-open without new evidence (this claim's local layer)

Empty at intake (2026-08-05). Nothing has been refuted in this thread yet.
The binding set for D1 is the union of `CLAIM_2D_BIAS_20260730.md`'s list and
`BIAS_HISTORY_LEDGER.md` §2, as checked above.
