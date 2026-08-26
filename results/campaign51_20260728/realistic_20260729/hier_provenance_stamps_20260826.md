# [HIER] provenance stamps + rule-1 exoneration check — 2026-08-26

**Scope.** Author of task: orchestrator (Fable), tag `[OPUS-ORCH 2026-08-26]`. This file supports
the `[HIER]` (h,θ)-self-calibration prereg (D1–D7). It does two things per `[A11]` rule 12: (1) the
rule-1 exoneration check against ledger §2 + local `CLAIM_*.md` files; (2) a provenance stamp
{value, source, date, configuration-of-record, FRESH/STALE} for every quantity the prereg would
otherwise quote as a point number. Format follows `CLAIM_P3_MKER_20260826.md` §4 (the delimitation
house model: quote the exoneration text, then state the delimitation).

---

## 1. Rule-1 exoneration check — does the (h,θ) grid re-open a standing exoneration?

Read: `gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 (`:127–169`, "DO NOT RE-TRY") and §3
(`:172–207`, "HISTORY vs CURRENT CLAIMS"), plus every local `CLAIM_*.md`
(`CLAIM_2D_BIAS_20260730.md`, `CLAIM_P3_MKER_20260826.md`, `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`,
`CLAIM_B0_FINITE_MOMENT_20260824.md`, `CLAIM_P3_2D_20260825.md`,
`CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`, `CLAIM_SYMMETRIC_SELECTION_INSERTION_20260818.md`,
`CLAIM_P3_RPHI_20260822.md`, `CLAIM_F0_SEL_20260825.md`, `CLAIM_P3_WBHZERO_20260825.md`,
`CLAIM_D1_P0WINDOW_20260805.md`, `m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md`).

### 1a. The one live-risk exoneration: **C7 / G2b — the host-z numerator weight `w_pop`**

**Quote (ledger `:197`, §3 per-claim table):**
> **C7** (host-z numerator weight `w_pop` omits p_det and φ_cat) | **CONTRADICTED by a ratified
> derivation.** G2b **CONFIRMED** `w_pop = (dV_c/dz)/(1+z)` as "the unique weight consistent with
> the project's own rate model and with every selection integral", **exactly h-independent**,
> reducing to the point kernel as σ_z→0. Adding p_det would break that h-independence (binding
> gate 6, `PRODUCTION-KERNEL-FIX-SCOPING:170-180`). Also: the deconvolution **over-corrects** at
> σ_z/z ~ O(1) (#68/#62), i.e. the sign of C7's proposed fix is the *opposite* of the measured
> failure mode. Note too that "numerator-only" kernel changes are the exonerated class (#37, #70)
> | `G2b_host_z_volume_prior.md:413-436`

This is the ledger's most-recent standing ruling specifically about the **host-z kernel's
functional form**. Two adjacent facts sharpen the risk surface:
- Ledger `:149` (§2 item 11): `volume_deconv` kernel h-dependence — **exactly h-invariant to 1e-15
  (#75)**.
- Ledger `:88` (#68) / `:82` (#62): the deconvolution's **over-correction at σ_z/z ~ O(1)** is
  itself a *measured, ratified* failure mode, not an open question — this campaign's own median
  σ_z/z sits in exactly that regime (§2 below).

**Delimitation (PASSED with scope, following the `[P3-MKER]` §4 house model).** G2b's ratified
claim is about the **functional FORM** of the population/rate weight `w_pop = (dV_c/dz)/(1+z)` —
it is the unique weight *given* a σ_z, and that uniqueness argument does not depend on what value
σ_z takes. Per D2/D3 (orchestrator decisions), θ's (b, s) transform acts on the **photo-z
measurement-error kernel's own parameters** (a bias curve + scatter on the *observed-z-given-true-z*
Gaussian that gets convolved against `w_pop`), not on `w_pop` itself. On that reading the (h,θ)
grid does **not** literally re-open G2b/C7's ratified uniqueness derivation.

**But this delimitation is conditional, not free, on two counts that the prereg must state
explicitly, not silently inherit:**
1. **D3 is the gate that makes the delimitation true.** If θ's `s` is built as (or collides with)
   a rescaling that also touches `w_pop`'s own σ_z-dependence (rather than only the separate
   measurement-error convolution), the grid *would* reopen G2b. D3 is already flagged
   PRE-LAUNCH BLOCKING for exactly this reason — this rule-1 check independently confirms D3's
   necessity from the exoneration side, not only the code-hygiene side.
2. **The over-correction finding (#68/#62) is adjacent, not identical, territory.** θ exploring
   larger assumed-σ_z values walks directly into the regime the ledger already measured as
   "deconvolution over-corrects at σ_z/z ~ O(1)." This is not the same claim (that finding is
   about the *current, fixed* kernel's own over-correction; θ instead treats σ_z as an unknown to
   be jointly inferred, which is a different question — "is the fixed kernel right" vs "what is
   the ensemble's best estimate of the kernel's own parameters"). But the prereg's write-up
   should name #68/#62 as adjacent prior art so a reviewer does not mistake the grid for a
   re-litigation of it.

**Verdict: no standing exoneration is reopened, CONDITIONAL on D3 being resolved as a genuinely
separate axis from `w_pop`.** If D3's pre-launch check finds `smear_sigma_z` and θ's `s` are the
*same* knob (not a generalization), the delimitation above needs re-derivation before the grid
runs, because the grid would then be sweeping the very quantity G2b ratified.

### 1b. Checked and cleared — no re-litigation risk

- **Ledger §2 item 16** (`:154`): `galaxy.py (1+z)³ σ_z` — the file `datamodels/galaxy.py` was
  deleted (commit `90bd40ee`, CLAUDE.md known-bug #9, "MOOT — file deleted"). Not touchable by a
  grid over the current production code.
- **Ledger row #95** (`:115`): "the idealized (iiib) venue's host-z kernel is δ-like" —
  **REFUTED — KERNEL-FINITE**; this is a refutation of a *different* (now-dead) claim, not an
  exoneration the grid could reopen; it is background confirming the kernel is already finite
  everywhere the grid would run.
- **Ledger §2 item 13** (`:151`, "Information starvation — OVERTURNED (#41/#52). Do not resurrect
  as an explanation") — checked against the F5 forecast's use of "information-starved" language.
  **These are different claims, not a collision**, but they use overlapping vocabulary and must
  not be conflated in the prereg text: #41/#52 is about the *explanation for the observed H0
  rail* ("starved of information" was overturned — row #52: "a property of prior-INCONSISTENT
  estimators, not of the data; consistency is the cure"). F5 (§2 below) is a *Fisher/RMSE-style
  forecast of achievable precision under a self-consistent (correctly-specified) estimator* — an
  orthogonal, forward-looking question, not a claim about why the current estimator is biased.
  **The prereg must not cite F5's "information-starved" language as if it were, or supported,
  the overturned #41/#52 claim.** State the distinction explicitly wherever F5 is invoked.
- **`CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md:87,119`** — explicitly lists "photo-z kernels" as a
  term the impostor-harness accounting **cannot** speak to (out of scope by the harness's own
  admission), not an exoneration of the kernel itself. No conflict.
- **`CLAIM_D1_P0WINDOW_20260805.md:366-372`** — cites the F5 forecast document itself as a
  self-consistent-closure asset; this is a citation, not an independent exoneration; its
  freshness is handled in §2 below.
- **Row #192** (ledger `:2880`, the `[HIER]` opening ruling itself) already scopes the thread:
  *"interpretation-layer coherence — shared photo-z error-model hyperparameters + shared latent z
  of overlapping candidates — NOT the LISA global-fit data-stream problem; events stay physically
  and measurement-independent."* This is the author's own standing scope fence, not an
  exoneration, but the prereg should restate it verbatim as its own scope statement (it already
  answers a class of "isn't this the same as X" reviewer questions before they're asked).
- **Ledger §2 items 1–15, 17** (mass kernel, mixture, `w_G`, hard clamp, `w_pop` tuning, `p_det`
  inside numerator, depth truncation, zero-host fallback, Ω_m, `L_comp`/`B_num`, `volume_deconv`
  h-invariance, p_det anchor, Ω_m/heliocentric/PV, `galaxy.py`, numerator-only cleans) — **none
  concern the photo-z kernel, host-z error model, or z-smearing** as their subject; checked and
  not implicated.

### 1c. Overall rule-1 verdict

**PASSED, with one conditional dependency.** The (h,θ) grid does not re-open any standing
exoneration outright. The one adjacent standing ruling (C7/G2b, host-z kernel's ratified
functional form) is **not** reopened as long as D3 confirms θ's `s` is a distinct axis from
`w_pop`'s σ_z-dependence — which is precisely why D3 is already registered PRE-LAUNCH BLOCKING.
This check adds an exoneration-side reason to that gate, not a new blocker.

---

## 2. Quantity-by-quantity provenance stamps

Format: **{value | source | date | configuration-of-record}** → **FRESH / STALE** verdict.

### 2.1 F5 — σ_z/σ_M precision forecast (the headline citation)

| field | value |
|---|---|
| **Value(s)** | Synthetic n(z), idealised: "with-BH-mass channel tolerates ~50× larger σ_z but only at σ_M ≲ 1–2%" (useful-5% σ_M boundary at GLADE photo-z: **σ_M ≈ 1.2–1.7%**); real-GLADE-n(z) robustness pass: **realistic mass-channel gain ~1–3×**, not ~60×; both channels rail at GLADE's actual σ_z≈0.035 "even σ_M=0.5% does not rescue it" |
| **Source** | `docs/SIGMA_Z_SIGMA_M_FORECAST.md` §2–4; engine `scripts/bridge_closure/sigma_z_sigma_M_forecast.py`; data `scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast.json` (config confirmed: `n_events=400`, `seeds=32`, `population=synthetic`, `sigma_dL_frac=0.05`, `sigma_Mz_frac=0.001`, `h_lo=0.5,h_hi=0.96,h_step=0.01`) and `..._realnz.json` |
| **Date / commit** | Analysis dated 2026-06-30 (doc header); committed `b10433ff`, 2026-07-01 12:14 +0200 |
| **Configuration-of-record at authoring time** | Self-consistent closure (estimator's kernel = generating kernel, unbiased by construction); **no candidate eligibility window at all** (sums the full continuous mock population, no truncation/membership step); linear-Gaussian host-mass kernel (not log-normal); mass-blind selection denominator (idealisation (a) in §4.2); `M_z` treated as an independent measurement (idealisation (b)); N_events=400 |
| **FRESH / STALE** | **STALE as a point number — must be carried as a band/structural relation, not quoted verbatim.** Two independent reasons, both already flagged inside the document itself and by the orchestrator: (1) **Self-flagged N mismatch** — §4 caveat 4: *"Absolute % is at N_events=400... must be re-quoted at the paper's adopted N"*; this campaign's actual configuration is not 400 (§2.4 below finds N=1588 in-campaign, not the symptom card's 40–200 either — see the discrepancy noted there). (2) **Structural drift since authoring** — production adopted the symmetric mass-filter window in `cf4f8a2a` (2026-08-25, ~2 months after F5), which F5's engine never modeled at all (no window of any kind); separately, `[P3-MKER]` (opened today, 2026-08-26) has an **open, unresolved** claim that the current production with-BH kernel omits the R&V15 intrinsic scatter entirely (kernel width dominated by σ_cond ~1e-8) — i.e. today's actual production σ_M-handling is *not* the idealised σ_M sweep F5 reports against. F5 answers "what precision would be needed," which remains the right question for the (h,θ) grid to be bound by (it is the *best-case*, correctly-specified-kernel information ceiling — misspecification only makes the achievable width worse, never better, so the ceiling direction of the bound is safe) — but its **point percentages** (1.2–1.7%, ~50×, ~1–3×) must not be quoted as current-configuration numbers without re-deriving them at this campaign's actual N and against the (still unresolved) `[P3-MKER]` kernel state. |

### 2.2 Median σ_z/z ≈ 49%

| field | value |
|---|---|
| **Value quoted in symptom card** | "median σ_z/z≈49%" (`STAGE_L_HIER_20260825.md:5`; ledger `:2884`, row #192) |
| **Closest measured source found** | `CLAIM_2D_BIAS_20260730.md:426`: *"the measured σ_z/z = 0.25–0.49 for these hosts"* (C7 mechanism, 76-host local sample); `:453`: *"at σ_z/z = 0.25–0.49 the inflation is +16% to +49%"*; `:462-463`: *"the production data independently implies σ_z/z ≈ 0.35–0.6"* |
| **Date** | 2026-07-30 (`CLAIM_2D_BIAS_20260730.md`, C7 adjudication, §6.6) |
| **Configuration-of-record** | 76-host local sample (stale `z_error` column, flagged `:437` as "indicative" not authoritative) for the 0.25–0.49 range; a *separate*, production-data-implied range of 0.35–0.6 from the ball-numerator tilt match |
| **FRESH / STALE** | **NOT FOUND as a direct "median" statistic.** Every underlying source found states a **range** (0.25–0.49 local-sample; 0.35–0.6 production-implied), never a single median value. "0.49" is the *upper bound* of the local-sample range, not a stated median of any distribution. This is a genuine transcription gap in the symptom card, not a measurement — **carry as the range (0.25–0.6 spanning both cited sources), not as a point "≈49% median,"** or re-derive an actual median from the campaign's per-event `z_error`/CRB sidecars before quoting one. |

### 2.3 z-structured score-at-truth tilt (≈0 below z≈0.4, ≈−1 by z≈0.9)

| field | value |
|---|---|
| **Value** | "score ≈ 0 below z ≈ 0.4, monotone to **−1.08 at z ≈ 0.9**" |
| **Source** | Ledger `:1352` (row #137, item 3, "Localization"), citing `PREREG_COMPLETION_CLASS_DECOMPOSITION.md`; aggregate figure it localizes: dark-class score at truth **−0.635 ± 0.017** (iiib, 37σ) / −0.565 ± 0.020 (joint_r1, 28σ), ledger `:1347-1348` |
| **Date** | 2026-08-20 (row #137) |
| **Configuration-of-record** | **Scoped to the pure-completion/dark class only** (605/1588 iiib events, 491/1588 joint_r1 — ledger `:1342`), *not* the full event sample. Row #137 item 1 explicitly states 2D is identical to 1D for this quantity ("2D identical (C-C 0.6004) ⇒ the base tilt is NOT mass-channel structure") — i.e. mass-channel/with-BH kernel state is not a confound for this specific number. |
| **FRESH / STALE** | **Aggregate (−0.635±0.017) confirmed FRESH** — ledger `:1647-1648` and `:1687-1690` (row #145/#146 addendum) explicitly re-verify: *"Post-fix baselines, dark-class 0.6001, score −0.635 ± 0.017: all untouched"* by the sentinel-defect fix, and confirm the mirror-only `g_frac=NaN` defect never reached production (0 zero-cells across all 5 production diagnostics runs checked). **The z-BINNED breakdown (0 below 0.4, −1.08 at 0.9) itself was not separately re-run post-fix by name** — only the aggregate was explicitly re-quoted — but since it is the same production event-level score distribution just re-binned by z, and the aggregate is confirmed untouched, contamination risk is low. Separately: row #137 (2026-08-20) **predates** the symmetric mass-filter window (`cf4f8a2a`, 2026-08-25); item 1's own finding that this tilt is mass-channel-independent makes the window unlikely to move it, but this has not been explicitly re-measured post-window-adoption. **Verdict: usable with two disclosed caveats** — (i) it is a dark/completion-class-only statistic, not a whole-sample one, and the symptom card's compression to "the per-event score-at-truth tilt" without that qualifier should be corrected in the prereg; (ii) not re-verified post-`cf4f8a2a`, though low-risk by the class-independence finding already on record. |

### 2.4 p_det ≈ 1

| field | value |
|---|---|
| **Value** | "per-event selection p_det ≈ 1 in the relevant regime" |
| **Source found** | Ledger `:56` (row #40, 06-30): *"no method validates σ_z/z≈0.7, z≈0.05, p_det≈1"* (`BRA:166-199`, i.e. `BIAS_RESOLUTION_ATTEMPTS_REPORT.md`) — a regime description from the June-era investigation, not a per-campaign measured statistic |
| **Date** | 2026-06-30 |
| **Configuration-of-record** | Not stated beyond "the relevant regime" (low z, GLADE-hosted, above SNR threshold) |
| **FRESH / STALE** | **NOT FOUND as a direct measurement within this campaign** (`realistic_20260729`). The only citation located is a June-era regime descriptor from a different investigation (`BRA`), not a number re-derived against this campaign's actual per-event `p_det` column. Consistent with production architecture (`SimulationDetectionProbability`, GLADE hosts, `SNR_THRESHOLD=20` — CLAUDE.md) but **carry as an assumption to be spot-checked against this campaign's own diagnostics CSV, not as a measured quantity of record.** |

### 2.5 N ≈ 40–200 events

| field | value |
|---|---|
| **Value** | "ensemble H₀ from N≈40–200 standard-siren events" |
| **Source found** | Only `STAGE_L_HIER_20260825.md:4,89,164` and ledger `:2883` (row #192) — i.e. the symptom card **itself**, no deeper measured artifact located anywhere in this campaign or the wider repo (`grep` for "40-200"/"40–200" outside `realistic_20260729` returns nothing) |
| **Date** | 2026-08-25 (symptom-card authoring) |
| **Configuration-of-record** | Unstated in the symptom card; **does not match this campaign's actual N**. `CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md:101,113` states the production catalogue-mode run count is **n_events = 1588** for this campaign's fusion/counterfactual work, an order of magnitude above "40–200." |
| **FRESH / STALE** | **NOT FOUND / UNSOURCED, and in tension with the campaign's own recorded N.** No file in this repo derives "40–200" from a measurement; it reads as an unattributed characterization written directly into the symptom card. The prereg must **not** quote it as a measured campaign parameter — either re-derive the intended N (e.g. a smaller "host-associated, well-localized real-mission" subset distinct from the 1588-event mock realization pool, if that is the intended referent) with its own citation, or use the campaign's actual N=1588 and say so explicitly. |

---

## 3. Quantities that may NOT be quoted as point numbers in the [HIER] prereg

Per §2 above, the following must be carried as **bands / ranges / re-derived numbers**, never as
the point values currently circulating:

1. **F5's σ_M ≲ 1–2% / ~50× / ~1–3× headline numbers** — STALE at N=400 and pre-`cf4f8a2a`;
   re-derive at this campaign's actual N and note the still-open `[P3-MKER]` kernel-width gap
   before quoting. The **qualitative** conclusions (frontier `σ_M·(1+z) ≲ σ_z`; GLADE rails
   regardless of σ_M at realistic scatter) remain the right bound to cite; the percentages do not.
2. **"Median σ_z/z ≈ 49%"** — no source states a median; only ranges (0.25–0.49 local-sample,
   0.35–0.6 production-implied) exist. Quote the range, or compute an actual median first.
3. **"N ≈ 40–200 events"** — unsourced anywhere in the repo; conflicts with this campaign's
   recorded N=1588. Do not quote without a fresh derivation or an explicit note that it is a
   different (unspecified) referent from this campaign's realized event count.
4. **"p_det ≈ 1"** — sourced only to a June-era regime descriptor (`BRA`, row #40), not a
   per-campaign measurement. Usable as a working assumption, not as a stamped campaign quantity,
   until spot-checked against this campaign's own p_det column.
5. **The z-binned tilt (0 below z≈0.4, −1.08 by z≈0.9)** — usable, but only with its true scope
   stated (dark/completion class only, 605/1588 and 491/1588 of events, not the whole sample) and
   flagged as not independently re-verified after `cf4f8a2a` (low risk, per row #137's own
   mass-channel-independence finding, but undisclosed if silently generalized).

The aggregate dark-class score-at-truth **−0.635 ± 0.017** (§2.3) is the one number in this table
confirmed FRESH by an explicit post-fix re-verification and may be quoted as-is with its class
scope stated.
