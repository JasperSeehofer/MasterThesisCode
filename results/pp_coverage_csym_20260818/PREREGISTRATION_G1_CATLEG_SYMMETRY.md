# PRE-REGISTRATION — G-1: catalogue-leg symmetry cells ([C-SYM]/[P3] front, harness scale)

**Date:** 2026-08-18 · **Status:** DRAFT v1 — awaiting verifier pre-check. **Execution
authorization:** the author's 2026-08-18 in-session directive (verbatim: *"Please move through
this research cycle autonomously and flag it accordingly. Make sure you dont make the same
mistake as last cycle where one of the instrument arms was void in the end. looking forward to
see the results!"*) grants autonomous execution of this cycle's [DO]-class steps
(orchestrator-derived reading, flagged for veto: freeze, build, preflight, run, readout);
**every branch call in §4 still returns to the author as a [RULE], presented not adjudicated.**
**Append-only after commit: no edits above the VERDICT line once this file is committed. Any
later edit to §7 voids the registration (new prereg required).**

**Provenance gating (what upstream gate makes this necessary):** row #121 item 3 (the author's
directive: collect the paper-grounding measurements once the pipeline is settled — the prodcal
verdict, G-4, is now banked, rows #122–#124); row #122 item 5 (the [C-SYM] claim admitted to
the record, stage-0 intake:
`results/campaign51_20260728/realistic_20260729/CLAIM_SYMMETRIC_SELECTION_INSERTION_20260818.md`);
row #124 item 3 (the [C-SYM]/[P3] front carried open with G-1/G-2 as grounding, each its own
[DO], prereg-first); the Gray-convention proposal §6 row G-1
(`docs/PROPOSAL_GRAY_CONVENTION_PAPER_INTEGRATION_20260817.md`). Registered purpose: G-1 is the
**harness-scale first stage** of the catalogue-leg fusion counterfactual — it decides whether
the ~170 CPU-h production protocol is warranted, and it converts the prodcal audit's ad-hoc
exploratory symmetric/catalogue-only numbers (+0.008 / +0.041 — produced by an uncommitted
in-session patch, **not reproducible from any committed artifact**; recon 2026-08-18) into
registered, verdict-bearing measurements.

**Exoneration-layer check (stage-0, hard rule 1):** nearest neighbours in the §2 DO-NOT-RE-TRY
list are item 6 ("p_det inside the numerator alone — refuted, breaks calibrated controls,
#66/#67") and item 2 ("full Gray mixture as compensation channel — amplifies, #60"). Neither is
re-opened: the asymmetric-insertion validity boundary was measured and mechanism-owned by rows
#120–#124 (safe flat-S̄ / unsafe strong-gradient, first-order tilt, 3% agreement), and [C-SYM]
was admitted by author [RULE] on that new evidence. This prereg proposes **no production
change** on any branch; it measures the correct-form question at harness fidelity. The standing
venue-scoping rule binds every read below in both directions.

**Stage-L note (R0 sweep, 2026-08-18, mandatory at stage 0):** nothing new beyond the L0-LIT
read. Two "no answer found in R0" results are on the record: (i) the
data-deterministic/latent-thresholded fork is this thread's own derivation, absent from the
cited papers; (ii) no cited paper states a σ_z→0ﾠvalidity condition for this bias class (G-2's
business). The MFG 2019 consistency-principle quote in
`docs/derivations/fixb_pathA_phi_marginal_selection.md` §1 is a repo paraphrase, not
verbatim-verified — registered in `docs/LITERATURE_WARNINGS.md` (row MFG-a, this session).

## 0. Instrument — the selection_cell extension (frozen in the same commit as this file)

The [A3]-extended `darksiren_emri/validation/pp_coverage.py` currently dispatches
`SELECTION_CELLS = ("off", "1d", "2d", "fused")` (`pp_coverage.py:1566,2006-2010`). This
registration extends it by two values, semantics registered here:

- **`cat1d`** — per-candidate catalogue-leg insertion only: the survival factor S̄_φ(z;h) (the
  SAME table object the completion leg uses) enters each catalogue candidate's 1D numerator
  term inside that candidate's z-kernel integral. No completion-leg insertion. No new
  normalization is introduced (per-leg kernel renormalization was staged and REFUTED,
  FULL-C/E ≈ −1100 nats/h overshoot — proposal §1.3); the normalization treatment mirrors the
  completion leg's registered form.
- **`symmetric`** — the [C-SYM] both-legs form: `fused` (landed [P2]+[P1]) PLUS the `cat1d`
  catalogue-leg factor. This is the claim's "S̄_φ in BOTH numerator legs, paired against the
  same α(h)".

**Scope limit (registered):** the extension is 1D-leg only. The catalogue-leg 2D per-candidate
fusion (per-galaxy single-∫dM with S_4D — [P3]'s full form) is NOT built; warrant: S-1
localized the venue's entire shift to the 1D completion insertion (2D inert, −0.0011), and the
production counterfactual measured the 2D channel near-inert (M-1). G-1 therefore bounds the
**1D-leg** [P3] materiality; the 2D-leg counterpart is a carried validity limit (§6).

**Builder ambiguity resolutions** (e.g. the exact pinning of S̄_φ inside the candidate
z-integral) are documented in the build note and are part of the instrument under test
(prodcal CC-3 convention); a FAIL traced to one is an instrument finding.

**Required tests (same commit):** (t1) byte-identity of all four pre-existing modes against the
prodcal freeze (`39e016d2` behaviour) — any diff blocks registration; (t2) S̄ ≡ 1 limit
(d50 → ∞): `symmetric` ≡ `fused` and `cat1d` ≡ `off` bit-exact; (t3) empty-catalogue-ball
limit: `symmetric` ≡ `fused` and `cat1d` ≡ `off` bit-exact; (t4) generative-stream alignment:
one master seed produces identical generative draws across all six selection_cell values;
(t5) α_M = 0 mass-blind reduction remains bit-exact in the new modes.

This is **instrumentation** (validation-harness code, not a physics-trigger file): plain GSD,
no `/physics-change` gate. Any production adoption of the symmetric form would be a separate
`/physics-change` proposal and is NOT proposed by any branch of this prereg.

## 1. Hypotheses (registered BEFORE any cell is run)

- **H-G1 (primary — the [P3] production-warrant decision).** At the production-analog flat-S̄
  venue V-prod, the catalogue-leg insertion is **immaterial on top of the landed form**: the
  paired per-realization MAP delta (symmetric − fused) is null at every truth (§4 band,
  two-sided). Registered meaning of PASS: the ~170 CPU-h production catalogue-leg
  counterfactual is NOT warranted; presentation option (a) carries a measured harness bound.
  Registered meaning of FAIL: the production counterfactual returns to the author as a fresh
  [DO].
- **H-SYM (the [C-SYM] verdict-bearing read).** At V-deep — the strong-gradient venue where the
  asymmetric `fused` form is measured confidently-wrong (−0.034, cov68 0.050 at n=250) — the
  `symmetric` cell **calibrates**: absolute bias and coverage within the §4 band at every
  truth. This registers the audit's exploratory +0.008 as a measurement. A FAIL here fires the
  [C-SYM] claim's Refute-by class (the claim named the V-flat symmetric variant; V-deep is the
  amplitude venue — a strictly stronger test of the same clause: if the asymmetry — not the
  insertion — owns the displacement, the symmetric form must calibrate where the amplitude is
  largest).
- **H-CAT (mechanism, secondary).** At V-deep, the catalogue-only insertion displaces
  **positive** (the audit's mechanism table: catalogue-only +0.041-class, opposite sign to
  completion-only −0.029): paired delta (cat1d − off) ≥ +0.010 coherent at h = 0.62 and 0.72.
  Point-prediction two-sidedness: a delta > +0.10 (≈ 2.4× the ad-hoc value) is
  instrument-suspect, not confirmation. *(AMENDMENT G1-4 iii.)*
- **Expected NULLs (registered):**
  - **N-A (byte-identity / instrument continuity):** the re-run `off` cells (both venues, the
    prodcal seeds) reproduce the on-disk prodcal cells **bit-exactly** (same harness modes,
    extended-knob commit). Any diff ⇒ STOP before scoring. **Registered disambiguation (the
    V-deep referent is cross-environment — cluster job 6355028 vs the dev machine — while the
    V-prod referent is local-vs-local):** if REP-OFF-D differs while REP-OFF-P is bit-exact,
    rerun the V-deep `off` cell once at the pre-extension freeze (`fe72d52b` chain) on the same
    dev machine. If that rerun ALSO differs from the on-disk cluster cell, the difference is
    environmental, not an instrument break: the local frozen-commit rerun becomes the N-A
    byte-identity referent, and the paired twins for P1 and the H-SYM/H-CAT context must then
    be locally rerun cells (same seeds; ≤ 0.4 CPU-h added, inside the ceiling), not the on-disk
    cluster cells. Only a diff that survives this environment control is scored as "extension
    broke the instrument": STOP, audit. *(AMENDMENT G1-2, applied verbatim.)*
  - **N-B (engagement):** every registered paired delta (§3) is non-degenerate (not identically
    zero), and the per-realization catalogue-bearing event fraction is > 0.5 at both venues
    (else `symmetric` cannot engage). Degenerate = silently-inert knob: STOP as N-A.
  - **N-C (registered non-read):** V-prod **absolute** legs are NOT scored — the raised-d50
    venue bias (VERDICT-3: off twin carries +0.007…+0.012 on its own) confounds them; only
    paired reads carry meaning at V-prod. This is a registered scope exclusion, not a null.

## 2. Stage-1 information forecast / power (anchored on measured prodcal scatter, no Fisher leg)

- Paired per-realization delta SE at n=250, R=120, shared stream (measured, VERDICT-2/3 class):
  **1.4–1.7e-4** (V-prod fused−off: ±1.57e-4/±1.39e-4/±0.33e-4; V-flat: ±1.7e-4/±1.4e-4).
  The symmetric−fused pair engages only the ~60–67% catalogue-bearing events; registered guard
  (**scoped to P4 / H-G1 only** — the V-deep pairs P1–P3 carry the measured V-deep paired-SE
  class 7.1–8.1e-4 at n=250 and are scored against their own 10×-larger edges): the P4 read is
  scored only if realized paired SE ≤ 3.0e-4, else UNDETERMINED-BY-NOISE with the designated
  next measurement = the same pair at n=800 (returns as an amendment, ~1.1 CPU-h).
  *(AMENDMENT G1-3, applied verbatim.)*
- H-G1 materiality threshold 0.0018 (§5) is a **≥ 6σ** read even at the guard edge 3.0e-4; the
  PASS edge 0.0010 is ≥ 3.3σ. False-fail under the exact null (delta ≡ 0, SE 1.5e-4): the
  two-sided PASS leg |delta| ≤ max(0.0010, 2·SE) has false-fail ≪ 1% compounded over 3 truths
  (0.0010 ≈ 6.7σ).
- H-SYM discrimination: the two prior states are −0.034 (fused-like) and +0.008 (audit
  exploratory); absolute-bias SE at n=250, measured at this venue: 0.8–1.6e-3 (`off`
  0.00082–0.00133, `fused` 0.00020–0.00158) ⇒ the states are ≥ 26σ apart; the PASS edge 0.012
  sits ≥ 13σ from the fused state and ~2.5–5σ from the audit's +0.008 expectation (false-fail
  ≲ 2% compounded over 3 truths). cov68 binomial SE at R=120 = 0.043: PASS edge 0.50 is
  4.2σ below nominal 0.68 (false-fail ≪ 1%/truth); the fused state 0.050 is ~10σ below the edge.
- H-CAT: +0.041-class vs the measured V-deep paired-SE class 7–8e-4 — still ≥ 50σ, trivially
  powered; its risk is amplitude (UNVERIFIED ad-hoc source), carried via the two-sided band,
  not via power. *(AMENDMENT G1-4 i/ii, applied verbatim.)*

## 3. Design — cells, venues, seeds, scorer

**Venues (frozen from the prodcal registration; no new tuning discretion):**
- **V-deep:** z_support = 0.40, sky_frac = 1e-4 (`pretuning/CHOSEN.json`), default d50;
  n_galaxies = 2e5, mass ON (α_M = 0.25), catalogue_mode, kernel "volume", mixture "absolute".
- **V-prod:** as V-deep except z_support = 0.75, d50_gpc = 8×D50_GPC
  (`pretuning/CHOSEN_VPROD.json`; completion 0.384 on the 0.371 production anchor).

**Cells (all: n_events = 250, R = 120, truths {0.62, 0.72, 0.84}, h_step 0.004,
noise-model = production):**

| cell | venue | selection_cell | seed | role |
|---|---|---|---|---|
| REP-OFF-D | V-deep | off | 20270818 | N-A byte-identity vs `cells/vdeep_250_production_off` + paired twin |
| SYM-D | V-deep | symmetric | 20270818 | H-SYM absolute read; P1/P2 |
| CAT-D | V-deep | cat1d | 20270818 | H-CAT; P3 |
| REP-OFF-P | V-prod | off | 20271218 | N-A byte-identity vs `vprod_250_production_off` + context twin |
| SYM-P | V-prod | symmetric | 20271218 | H-G1 decision read; P4/P5 |

**Deliberate seed reuse (flagged, Block-N1 convention):** 20270818 and 20271218 are the prodcal
Block-A/AMENDMENT-4 master seeds — reused ON PURPOSE so the existing on-disk `fused` and `off`
cells (`results/pp_coverage_prodcal_20260817/cells/`) are stream-paired twins at zero marginal
cost (rule [A1]) and so N-A is a run-level byte-identity check on the knob extension. No fresh
generative venue is being sampled; the registered reads are paired, which is exactly what the
shared stream buys.

**Registered pairs (PAIRS manifest, scorer pre-committed in the freeze commit):**
P1 = SYM-D − fused(prodcal `vdeep_250_production_fused`) · P2 = SYM-D − REP-OFF-D ·
P3 = CAT-D − REP-OFF-D · P4 = SYM-P − fused(prodcal `vprod_250_production_fused`) ·
P5 = SYM-P − REP-OFF-P (context). Scorer invocation of record: `readout_g1.py --registered
<cells_dir>`; per cell × truth: MAP bias mean ± SE, cov50/68/90 ± binomial SE, rail fraction;
per pair: per-realization delta mean ± SE, quartiles, degeneracy flag ([A2]: every cross-cell
comparison is reported as the paired distribution alongside the class mean, never the aggregate
alone). No statistic outside the scorer enters the verdict. **Path resolution (AMENDMENT
G1-7):** the P1/P4 reference cells live in `results/pp_coverage_prodcal_20260817/cells/`; the
`readout_g1.py` PAIRS manifest carries explicit per-cell paths (via `--prodcal-cells-dir`) so
`--registered <cells_dir>` is a complete invocation.

**Budget (measured prodcal cell times):** REP-OFF-D 416 s + SYM-D ≈ 1200 s + CAT-D ≈ 1000 s +
REP-OFF-P 305 s + SYM-P ≈ 770 s ≈ **1.0 CPU-h**; ceiling **3 CPU-h** (symmetric-cell cost is
an estimate — 1.2× fused — not a measurement). Local dev machine, single core per cell.
Exceeding the ceiling stops execution and returns to the author.

## 3b. Arm-validity preflight (registered, non-scored — the anti-void gate)

Author-mandated this session (the V-ctrl lesson: an arm ran 120 realizations with
completion_fraction ≡ 0 because engagement was never checked before running). **Before any
scored realization, every registered cell runs an R = 4 probe at its exact registered
configuration** (same seed base, probe-flagged) and must show:

1. completion_fraction ∈ [0.05, 0.95] and catalogue-bearing event fraction > 0.3 at both
   venues (V-deep anchor 0.6705/0.3295 — `pretuning/CHOSEN.json`; V-prod: completion 0.384 —
   `CHOSEN_VPROD.json`, which carries no host-in-ball field; realized catalogue-bearing
   fraction 0.626 at R=120 — `vprod_250_production_off` diagnostics) *(AMENDMENT G1-5)*;
2. the probe's paired delta distribution non-degenerate for every registered pair P1–P5 (the
   symmetric and cat1d levers demonstrably act before 120 realizations are spent);
3. finite likelihoods, no NaN/rail-pinned probe MAPs at any truth;
4. S̄_φ table support covers the completion window AND the catalogue candidates' kernel range.

Any violation ⇒ **STOP before the scored run**, diagnose, return as an amendment. Probe outputs
archived in `preflight/`, never scored, never quoted as measurements. The N-A byte-identity
comparison additionally runs at probe scale first (R=4 arrays bit-compared) so a broken stream
is caught before the full replication cells run.

## 4. Falsifiable bands (registered)

Per truth; paired reads on the shared stream; "delta" = per-realization paired MAP delta mean.
**All registered reads (H-G1, H-SYM, H-CAT, N-A, N-B) are scored on the 1D channel
(`channel_1d`) only; 2D-channel statistics are reported descriptively and are never
verdict-bearing** — the V-deep 2D channel carries a venue-intrinsic +0.008…+0.010 bias with
cov68 0.375–0.483 in the `off` twin itself (the §9-flag-3 noise-coupling class, present with
the fusion off), so no absolute 2D band is satisfiable at this venue. *(AMENDMENT G1-1,
verifier pre-check, applied verbatim before freeze.)*

| read | PASS | FAIL | MIXED (first-class) |
|---|---|---|---|
| **H-G1**, P4 (symmetric − fused), V-prod | \|delta\| ≤ max(0.0010, 2·paired-SE) at every truth (two-sided null band) | coherent-sign \|delta\| ≥ 0.0018 at ≥ 2 truths (§5 materiality) | else — incl. sub-materiality but ≥ 3σ structure: quantify vs §5. Guard: any truth's paired SE > 3.0e-4 ⇒ UNDETERMINED-BY-NOISE (designated next: n=800 amendment) |
| **H-SYM**, SYM-D absolute legs, V-deep | \|bias\| ≤ 0.012 AND cov68 ≥ 0.50 at every truth | coherent \|bias\| ≥ 0.020 at ≥ 2 truths OR cov68 ≤ 0.20 at ≥ 2 truths (the fused-like state ⇒ fires [C-SYM]'s Refute-by class) | else (e.g. partial restoration) |
| **H-CAT**, P3 (cat1d − off), V-deep | delta ≥ +0.010 coherent at h = 0.62 and 0.72 AND delta ≤ +0.10 everywhere | delta ≤ +0.005 at h = 0.72, or negative coherent sign (mechanism table refuted) | else; delta > +0.10 anywhere = instrument-suspect MIXED (STOP-class, audit before use) |
| **N-A**, REP-OFF-{D,P} vs on-disk prodcal cells | bit-exact (MAP arrays byte-identical) | any difference (instrument suspect: STOP) | — |
| **N-B**, engagement | all P1–P5 non-degenerate; catalogue-bearing fraction > 0.5 both venues | any degenerate registered pair (STOP as N-A) | fraction ∈ [0.3, 0.5]: scored but flagged |

**Band derivations (A8-v2 disclosure):** H-G1 PASS edge 0.0010 ≈ 6.7σ at SE 1.5e-4 (false-fail
≪ 1% over 3 truths); FAIL edge 0.0018 = the §5 materiality yardstick (⅓ × 0.0053). H-SYM PASS
edges: bias 0.012 ≈ 1.5× the audit's exploratory +0.008 (which is ad-hoc/UNVERIFIED — the band
is deliberately generous against it) and ≥ 13σ from the fused state at the measured absolute
SE class ≤ 1.6e-3 *(AMENDMENT G1-4 iv)*; cov edge 0.50
= 4.2σ below nominal at binomial SE 0.043 (false-fail ≪ 1%/truth; ~1% over 3 truths). H-CAT
edges: +0.010 ≈ ¼ of the ad-hoc value; +0.10 = an order beyond it. N-A is exact by
construction (t1/t4 tests make bit-identity the designed behaviour).

**Branch calls (registered; every branch returns to the author as a [RULE] with the data —
nothing is adjudicated here):**
- **H-G1 PASS + H-SYM PASS:** proposed reading — [C-SYM]'s correct form is confirmed at the
  amplitude venue AND measured immaterial in the production-analog regime; the production
  catalogue-leg counterfactual is NOT warranted; presentation option (a) becomes the honest
  pick with this measured bound, option (b) remains available as a derivation-first
  presentation. The (a)/(b) pick is the author's [RULE] with G-2 alongside.
- **H-G1 FAIL:** the production counterfactual (~170 CPU-h class, item-4 protocol) returns as a
  fresh [DO] with the measured harness delta as its sizing input.
- **H-SYM FAIL:** [C-SYM]'s Refute-by class fires — the asymmetry-ownership mechanism is
  contradicted at the amplitude venue; the claim returns to stage 0 with the new data; the
  paper's §1.2 framing must not present the symmetric form as calibration-restoring.
- **H-G1 PASS + H-SYM MIXED (partial restoration):** first-class MIXED; designated separating
  cell: the V-flat symmetric cell (flat-S̄, the claim's original Refute-by venue, ~0.2 CPU-h)
  returns as an amendment.
- **H-CAT FAIL** (delta ≤ +0.005 at h = 0.72 or negative coherent sign): the audit's
  mechanism table (catalogue-only +0.041, opposite sign to completion-only) is contradicted
  at registered fidelity; returns to the author as a [RULE] annotated onto the [C-SYM]
  intake card, and any H-SYM interpretation that leans on the two-leg sign-cancellation
  story is barred until re-derived. *(AMENDMENT G1-6, applied verbatim.)*
- **Execution-completeness (A8-v2 d):** no count-based branch here is adjudicable while any
  registered cell is unrun; all five cells run before any branch is called.

## 5. Materiality yardstick (registered before data)

|delta| ≥ ⅓ × the narrower of (i) the production campaign posterior widths of record
(σ = 0.0053, iiib, row #119 context) and (ii) the F5 forecast width at the campaign venue ⇒
**material** ⇒ threshold 0.0018. The comparison is a computation; its consequence is the
author's [RULE].

## 6. Carried caveats and validity limits (registered)

1. **Venue transfer:** harness fidelity only; a PASS/FAIL here is not a production
   certificate/indictment; the venue-scoping rule binds both ways. The production-warrant
   decision (H-G1) is explicitly a harness-scale screen — that is its registered role.
2. **1D-leg scope:** the 2D-leg per-candidate fusion ([P3] full form) is unbuilt and unmeasured
   here; H-G1 bounds the 1D-leg materiality only (warrant in §0).
3. **Ad-hoc prior values:** +0.008/+0.041 come from an uncommitted in-session audit patch —
   prior expectations, not instrument output; the bands above do not assume them beyond edge
   placement (disclosed in §4).
4. **V-prod absolute legs venue-confounded** (raised-d50 bias, VERDICT-3) — registered
   non-read (N-C).
5. **CC-1..CC-3 of the prodcal intake** carried unchanged; builder resolutions are part of the
   instrument under test.
6. **No production change is proposed by any branch.** Adoption of the symmetric form in
   production would require its own `/physics-change` gate and reconciliation with the [P3]
   paper presentation ruling.
7. **2D-channel absolute statistics are venue-confounded at V-deep** (the +0.01-class
   noise-coupling bias, present identically in `off`); they are reported descriptively only
   (mirrors G-2 caveat 5). *(AMENDMENT G1-1.)*

## 7. Execution appendix — filled at freeze, before any scored run

- **Instrument hash of record:** the freeze commit carrying this file (the harness extension,
  `test_pp_coverage_csym.py`, the scorers/drivers and this prereg are frozen in the SAME
  commit — the commit IS the hash of record, prodcal convention).
- **Test record at freeze:** `darksiren_emri_test/validation/` 290 passed / 2 deselected
  (t1–t5 in `test_pp_coverage_csym.py`, 13 tests); mypy + ruff clean on the changed files;
  whole-tree fast suite run pre-commit (result in the freeze-commit session log).
- **Invocations of record:** cells via `uv run python run_g1.py` (this directory; A-PF-5
  order: N-A cells first, then science cells; per-cell JSON into `cells/`, idempotent);
  preflight via `uv run python run_g1.py --preflight` (R=4, engagement/rails only —
  executed pre-freeze, READY, outputs in `preflight/`); scoring via
  `uv run python readout_g1.py --registered cells/ --referent-dir cells/
  --prodcal-cells-dir ../pp_coverage_prodcal_20260817/cells/` → `readout_g1_output.json`.
- **Environment:** local dev machine, single core per cell, sequential (frozen-seed serial
  streams); N-A referent `cells/referent_preext_vdeep_250_production_off.json` computed
  pre-freeze from a clean worktree at pre-extension HEAD `6504c8b9` [LOCAL].
- No pretuning stage exists in this registration (venues frozen from prodcal CHOSEN files —
  no tuning discretion).

---

## PRE-FREEZE AMENDMENT A — 2026-08-18 — grid headroom + all-local twins (responsive to the
## registered §3b preflight and the builder's N-A environment control; applied before freeze)

**Trigger 1 (G-2 preflight STOP, mechanism-consistent):** the `cat1d` probe rails at
h_true = 0.84 (R=4, all probe MAPs at the grid edge h_max = 0.86) — the +0.04-class positive
catalogue-leg displacement pushes the top truth off-grid. Unfixed, every cat1d/symmetric read
at h = 0.84 would be a grid-edge void arm (the class this cycle is mandated to prevent).
**Trigger 2 (N-A environment control, executed per AMENDMENT G1-2):** with every extension
edit reverted, a fresh local run of the unmodified harness at the registered V-deep config/seed
diverges from the on-disk cluster cell at 2 of 3 truths — the cluster-vs-local diff is
**environmental**, so per G1-2's registered disambiguation the N-A referent and the P1/P4
paired twins migrate to locally-run cells.

**Amended design (supersedes the §3 cell table; venues, seeds, truths, R, noise unchanged):**

1. **Estimator h-grid for all science cells:** extended to **h ∈ [0.56, 0.92]**, h_step 0.004
   unchanged, extension by whole steps so interior grid points stay aligned with the prodcal
   grid. Headroom covers ±0.04-class displacements at both edge truths (per-trial map_std class
   0.011–0.015, measured V-deep n=250: P(MAP > 0.92 | center 0.881) ≈ 0.02–0.5% and
   P(MAP < 0.56 | center 0.586) ≈ 0.9–4.2% per trial — both under the 0.10 rail gate). Cost
   ≤ +38% per cell (node ratio 91/66; sub-linear in practice — ×1.29 estimated), ceiling
   holds at either figure. *(A-PF-2, applied verbatim.)*
2. **Science cells (wide grid, all run locally at the freeze commit):** V-deep seed 20270818:
   OFF-D-W, FUSED-D-W, SYM-D, CAT-D; V-prod seed 20271218: OFF-P-W, FUSED-P-W, SYM-P.
3. **Pairs (manifest updated accordingly):** P1 = SYM-D − FUSED-D-W · P2 = SYM-D − OFF-D-W ·
   P3 = CAT-D − OFF-D-W · P4 = SYM-P − FUSED-P-W · P5 = SYM-P − OFF-P-W. All twins are now
   same-environment, same-grid, same-seed — the strongest paired form available; the on-disk
   prodcal cells remain the §2 anchor source but are no longer paired referents.
4. **N-A (byte-identity) cells, original grid [0.60, 0.86]:** REP-OFF-D vs a local
   pre-extension rerun (clean worktree at the pre-freeze HEAD, same config/seed — the G1-2
   environment-control referent); REP-OFF-P vs the on-disk `vprod_250_production_off`
   (local-vs-local, as registered). Bit-exactness required as in §1 N-A.
5. **H-SYM/H-CAT absolute anchors:** scored against the wide-grid OFF-D-W/FUSED-D-W cells'
   realized values; §4 bands unchanged (band edges are venue properties, not grid properties;
   the prodcal-measured anchors in §2 remain the power calculation's source, now with the
   local off/fused cells as their in-campaign cross-check).
6. **Rail-fraction validity gate (registered):** any science cell × truth with 1D rail
   fraction > 0.10 on the WIDE grid is scored UNDETERMINED-BY-RAIL for absolute reads at that
   truth (paired reads report the rail fraction alongside); the fused V-deep floor-rail at
   h_true = 0.62 (prodcal-measured 99%-class at the old floor 0.60) is expected to relax on
   the wide grid — if FUSED-D-W still rails > 0.10 at 0.62, H-SYM/H-CAT sign-coherence reads
   at that truth carry the flag. **Precedence (registered, A-PF-1):** band legs quantified
   "at every truth" are evaluated over the non-rail-flagged truths only, and any PASS/FAIL
   adjudication requires ≥ 2 scoreable truths; a read with ≥ 2 truths UNDETERMINED-BY-RAIL is
   itself UNDETERMINED-BY-RAIL (unscored, returns to the author with the rail diagnostics). A
   rail-flagged truth never counts toward a "coherent at ≥ 2 truths" FAIL leg.
7. **Budget update:** science 7 cells ≈ (416+1000+1200+1000+305+640+770) s × 1.29 ≈ 1.9 CPU-h
   + N-A cells (416+305) s + worktree referent 416 s ≈ 0.3 CPU-h ⇒ total ≈ 2.2 CPU-h,
   ceiling 3 CPU-h unchanged.

Verifier one-item pre-check: recorded below before freeze. No band or hypothesis is changed by
this amendment; it changes instrument range, twin locality, and adds the rail validity gate.

**N-A comparison-scale clause (2026-08-18, pre-freeze; resolves the builder's reported
"local-vs-local N-A divergence"):** byte-identity is defined ONLY between equal-R runs at
identical config/seed — per-realization child seeds are drawn from one master stream through
the truths × R loop, so only the FIRST truth's first min(R, R′) realizations prefix-match
across different n_realizations and every later truth diverges by construction (the observed
2-of-3-truths divergence is exactly this signature); probe-scale (R=4) N-A comparisons
against R=120 cells are therefore VOID *(A-PF-6)* and are excluded from the preflight (the preflight checks engagement/rails only; N-A is scored at full
R=120). The G1-2 environment control, executed at FULL scale, in fact shows **cluster-vs-local
bit-exactness** (local pre-extension R=120 rerun vs on-disk cluster
`vdeep_250_production_off`: maps arrays bit-identical at all three truths, maxabsdiff 0.0
[LOCAL]; thread-count and extension-vs-HEAD invariance also verified bit-exact) — the
"environmental difference" reported at probe scale was a comparison artifact. Consequently the
on-disk prodcal cells ARE valid N-A referents; the worktree referent
(`cells/referent_preext_vdeep_250_production_off.json`) is retained as the pre-extension
control of record, and item 4's REP-OFF-D check scores against the on-disk cluster cell
`vdeep_250_production_off.json` as the referent of record, with
`referent_preext_vdeep_250_production_off.json` retained as the pre-extension control
(verified bit-identical to the referent, both channels, all truths) *(A-PF-7)*.
**Execution order (registered, A-PF-5):** REP-OFF-D and REP-OFF-P run and are scored for N-A
bit-exactness FIRST — before any wide-grid science cell launches (cost 416 + 305 s); an N-A
diff therefore stops the campaign at ≤ 0.2 CPU-h spent, preserving the early-catch role the
probe-scale comparison was registered to provide.
The trigger-2 paragraph above is superseded to this extent; the all-local wide-grid twins of
items 2–3 stand on the grid-change ground alone.

---

## VERDICT

*(append-only below this line after execution)*
