# CLAIM INTAKE — Production calibration harness front (#66/#67, pp_coverage mass channel)

**Date:** 2026-08-17 · **Research-cycle stage:** 0 (claim intake) · **Status:** PROPOSE-ONLY — no
source edited, no run launched, no ledger row until the author rules on §8.

**Provenance-gating (what upstream gate made this necessary):** the selection fusion landed in
production ([PHYSICS] commit `2b10b8b8`, rows #117–#118) explicitly *carrying* the #66/#67
calibration caveat as its named residual risk — "the single most likely way this correction
disappoints" (`docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md:71`; MINOR-4 in
`PROPOSAL_2D_SELECTION_FUSION_20260817.md:144`; honest gap 5 in the verifier addendum `:71`).
Row #119 item 3 carries it open: "#66/#67 production calibration (pp_coverage mass channel,
TO-BUILD) remains the stated disappointment path." Runbook 18 §1.2(a) names it the next front.

---

## 1. The claim under assessment

> **[C-CAL]** The landed fused estimator (`2b10b8b8`: [P2] S̄_φ(z;h) in the 1D completion
> numerator + [P1] S_4D fused inside `completion_mass_factor_g_sel`'s mass integral, dispatched
> `'auto'→'fused'` under `absolute_marginal`) **is calibrated at the production venue** —
> coverage-nominal and bias-free within forecast width — **including the mass channel, at
> production N, with multi-candidate host balls.**

Tag: **[INFER]** — inferred from the counterfactual's bounded deltas and the 1D-harness #67
result; **never measured**. No on-disk artifact can decide it (§5).

**Refute by (cheapest-first ladder, from stage-0 recon):**

1. **[R-1, cheapest]** Extend `catalogue_mode` pp_coverage with a minimal mass dimension and run a
   small-N P–P/coverage sweep exercising `completion_mass_factor_g_sel` directly — including the
   isolated `σ_cond→0` limiting cell, which the function's own docstring identifies as
   effectively the production operating point (d_L-conditional σ_cond p50 = 8.8e-8). Refuted if
   the mass-bearing cells show a coherent ≥1–2σ MAP displacement or coverage failure.
2. **[R-2]** Truth-injection coverage A/B (fused vs off) at intermediate N (200–500), mirroring
   the counterfactual's cell structure. Refuted if coverage fails under `fused` where it passed
   under `off` (or vice versa).
3. **[R-3, decisive]** The full [A3]-compliant harness: reproduce the #66/#67 pdetnum/noisemodel
   A/B ladder methodology at production N (1588) with mass channel + catalogue_mode. Refuted if
   the ~+0.0005 residual is N-coherent and scales material at production N.

---

## 2. What #66/#67 actually established (verbatim provenance)

Rows in `gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1, evidenced by the 2026-07-11 sweeps:

- **#66 — REFUTED: p_det-inside-numerator alone** (`pp_coverage_pdetnum_20260711/SUMMARY.md:12-22`):
  deep cells unchanged (Δbias ≤ +0.0006) while on untruncated controls the factor **flips** the
  small negative control bias (−0.0030…−0.0018 → +0.0028…+0.0063) and degrades cov68
  (0.675→0.550, 0.675→0.575). And `:19-22`: p_det-inside, "although it is the formally exact
  conditional for this latent-thresholded generative model — measures WORSE than the
  Mandel–Farr–Gair no-p_det-inside form on the calibrated controls." Ledger: "Do not cargo-cult it."
- **#67 — CONFIRMED dominant: the noise model** (`pp_coverage_noisemodel_20260711/SUMMARY.md:18`):
  applying BOTH (model-σ + p_det-inside) removes ~85–90% of the floor (+0.002…+0.005 → ≤+0.0008,
  cov68 restored at campaign-scale n). **Neither half alone works.** Const-σ floor is a **real
  asymptotic bias**: flat in n while cov68 collapses 0.63→0.38→0.12 at n=250/1000/4000
  (`SUMMARY.md:74,80-82`). A ~+0.0005 second-order residual survives the fully-consistent
  estimator.

**The intake-critical reading:** production has now landed the *selection-inside half* of the #67
pair. #67 says that half calibrates **only when paired with the consistent noise model**. Whether
production's noise treatment constitutes the required pairing has **not been checked** — that is
Q-0 in §6 and the sharpest cheap question this front owns.

## 3. Exoneration reconciliation (hard rule 1 — both layers checked)

Binding union read in full (claim-file list `CLAIM_2D_BIAS_20260730.md:721-756` + ledger §2
`:127-170`). Colliding entries: §2 item 6 ("p_det inside the numerator alone — refuted (#66);
only the joint pair works (#67)") and the claim file's unscoped "p_det inside/outside".

**This front does NOT re-open #66.** Reconciliation follows the four axes already worked out in
`CLAIM_N2_SELECTION_NUMERATOR_20260805.md.DRAFT` (draft-status, cited as analysis not authority):

- **Venue** (`:107-115`): #66/#67's venue is the 1D-only, single-effective-host, mass-free
  synthetic harness; the standing rule (`BIAS_HISTORY_LEDGER.md:150-152`) makes negative
  conclusions venue-scoped — binding both ways.
- **Object** (`:117-122`): #66 tested `p_det(d_L(z;h))` on a retained coordinate; production's 1D
  factor is `S̄_φ(z;h)`, the marginal over the *discarded* mass coordinate.
- **Channel** (`:124-127`): the 2D half (S_4D inside the mass integral) "has never been tested by
  anything."
- **Question type** (`:88-105`): #66/#67 adjudicated a bias-mechanism question, not estimator
  calibration correctness.

Carry-over caveats **CC-1..CC-3** (`:129-148`) are adopted as intake constraints: measurement-
gated before any further production change; completion-leg scope only (catalogue leg stays with
row #110); the blanket reading of "p_det inside/outside" is the author's call (§8, item R-1).

This front *calibrates the landed estimator*; it re-litigates nothing. If the harness build were
ever used to argue the fusion should be reverted, that returns to the author as a fresh [RULE].

## 4. Build target vs what exists ([A3] acceptance criteria)

`pp_coverage.py` today (1744 lines; entry `run_coverage`, `pp_coverage.py:1379`): switchable
host-z kernel, 5 mixture modes, and — since `44ee9125` (2026-07-26) — `catalogue_mode`: a
discrete shared frozen catalogue with hard sky caps producing genuine multi-candidate balls with
impostors.

| [A3] criterion | status | evidence |
|---|---|---|
| (i) 2-channel, completion-leg mass factor g recomputed per h, never frozen | **MISSING** | module docstring: "No mass dimension"; no g_sel analogue in file |
| (ii) production N (1588) | **MISSING** | census of all 9 `pp_coverage_*` dirs: n_events=250 everywhere except the #67 n-scaling arm (250/1000/4000); nothing near 1588 |
| (iii) multi-candidate host balls | **PARTIAL** | `catalogue_mode=True` provides it; row #86's "cannot exercise" caveat is now scoped: that run (`786dc8db`, `mixture_mode="absolute"`) predates `catalogue_mode` (`44ee9125`, +19h same day) and never used it |

Landed code paths the harness must be faithful to (all verified at current HEAD):
`completion_mass_factor_g_sel` (`bayesian_statistics.py:2155`), the [P2] insertion + dispatch
read (`:4496-4514`), `precompute_phi_marginal_survival` (`:1877-1951`), constructor
`selection_in_completion_numerator='auto'` validation/resolution (`:3010,3037-3047`), and the
`_G_SEL_S_VAR_TOL` quadrature-escalation guard.

## 5. [A1] zero-compute audit — why this front genuinely needs a build

All free re-reads enumerated and exhausted (stage-0 recon §5): the six 07-11 sweeps are 1D-only
and pre-fusion; the fusion counterfactual (`readout.json`, n_events 1588) is a **delta**
measurement (fused vs off) at production scale — it bounds what fusion *changed*, it cannot say
the result is *calibrated* (no truth injection, no coverage statistic). The per-event
`event_likelihoods.csv` dumps audit individual events, not coverage. **No existing artifact
decides [C-CAL].**

## 6. Stage-L R0 sweep — checks to fold in before/into pre-registration

Ranked (signature-match × cost) from the independent R0 sweep:

- **Q-0 (intake-owned, from §2): production noise-model pairing status.** Does the production
  estimator's distance-error treatment constitute the "model-σ" half that #67 says is mandatory
  for the selection-inside half to calibrate? Bounded code read of `bayesian_statistics.py` +
  CRB-CSV usage. **Measurement-before-gate rule 6: run before stage 2.**
- **Q-1: G23-c same-object check** (register row, UNCHECKED): is the fused selection object the
  *same* LOS-prior object the event term uses, so the h-dependent normalization cancels
  (Gray 2023 §2.1.4)? Code-level read against `2b10b8b8`. Low cost, highest match.
- **Q-2: G5b P3 re-check** — `docs/gates/G5b_chimera_icarogw_inspection.md` predates the fusion
  and is stale; re-ask its numerator-vs-selection-denominator same-model question against current
  code.
- **Q-3: Talts et al. 2018 / Cook–Gelman–Rubin 2006 own stated caveats** — cited as the harness's
  method (`calibration_gate.py:176-177`) but never interrogated; read for SBC's named failure
  modes (esp. blindness to generator/estimator-shared wrong terms — the D1-class complement that
  motivates keeping the absolute-count audit leg mandatory).
- **Q-4: Essick & Fishbach 2024 (2310.02017)** — promote from report-level mention to a
  quote-verified register row mapped onto the fused g_sel term.
- **Field-has-no-answer gaps (reportable):** no published precedent for dark-siren H₀
  coverage failure of the shared-filter shape; none for a mass second channel inside an SBC
  harness; none for multi-candidate marginalization as a calibration hazard. The harness must
  therefore *test*, not assume — and the build itself is publishable methodology.
- Five `docs/LITERATURE_WARNINGS.md` row updates proposed (text drafted in the sweep output;
  written only on [DO]).

## 7. Plan through the next stages (for approval, not execution)

- **Stage 0 exit** = this document + author ruling (§8).
- **Pre-stage-1 cheap checks:** Q-0..Q-2 (bounded code reads, no compute).
- **Stage 1 (information forecast):** what would a perfectly calibrated analysis of a
  mass-bearing synthetic venue say — F5-engine sizing of the coverage test's sensitivity, i.e.
  what Δbias/cov68 shift is *detectable* at each (n_events, n_realizations) so the [R-1]/[R-2]
  cells are powered, not decorative. (Fisher leg remains TO-BUILD per the known-gaps register;
  not needed for this sizing.)
- **Stage 2 (pre-registration):** A8-v2 discipline — numeric bands, first-class Mixed branch,
  expected nulls (e.g. `off` cells reproduce the 07-11 baselines; `σ_cond→0` cell matches the
  docstring limit `g_sel → g_i·S(μ_cond M_z,det)`), provenance-gating, append-only. Ladder
  [R-1]→[R-2]→[R-3]; [R-3]'s production-N cost estimated at prereg time from measured small-N
  runtime.
- **Stage 4 note:** both legs mandatory (coverage AND absolute detected-count audit) — SBC alone
  cannot catch a shared filter (stop/continue rule of record).

## 8. Decision table (author)

| # | tag | item |
|---|---|---|
| D-1 | **[DO]** | Open the front: accept this intake, authorize the pre-stage-1 cheap checks Q-0 (noise-model pairing status), Q-1 (G23-c same-object), Q-2 (G5b P3 re-check) as bounded read-only code audits. |
| D-2 | **[DO]** | Authorize the harness build to the [A3] spec (§4 gap list: mass channel + production-N capability on top of `catalogue_mode`), instrumentation-tagged (plain GSD — no physics formula changes; any formula need that emerges routes to `/physics-change`). |
| D-3 | **[RULE]** | Ratify the §3 exoneration reconciliation: this front calibrates the landed estimator and does not re-open #66/#67; ledger §2 item 6 and the claim file's "p_det inside/outside" entry remain binding as bias-mechanism verdicts, venue-scoped per the standing rule. (Resolves CC-3's blanket-reading ambiguity.) |
| D-4 | **[DO]** | Write the five proposed `docs/LITERATURE_WARNINGS.md` row updates (G23-c-check, Essick & Fishbach section, Talts/CGR section, G5b staleness flag, H-d promotion-on-instrument). |
| D-5 | **[DO]** | Stage-1 sizing + stage-2 pre-registration authored next (each returns to you before any run; [R-3]'s CPU budget filled at submission per row #116 discipline). |

Per the binding default, "approved" grants only the [DO]s; D-3 needs "ratified" (or equivalent)
to bind the record. Any verdict-dependent branch (e.g. what a failed [R-1] cell would imply for
the landed fusion) returns as a fresh [RULE].

---

*Sources: stage-0 recon dossier + R0 sweep + provenance follow-up (3 subagent reports,
2026-08-17, this session); all quotes verified at file:line as cited. Orchestrator-derived
itemization throughout; no author words are paraphrased as rulings.*
