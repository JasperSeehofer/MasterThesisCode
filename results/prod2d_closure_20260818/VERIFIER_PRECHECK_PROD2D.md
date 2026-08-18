# VERIFIER PRE-CHECK (Part VII) — Production-2D closure + landscape campaign

**Date:** 2026-08-19 · **File under review:**
`PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md` (DRAFT v1) · **Continues** Parts I–VI in
`results/pp_coverage_csym_20260818/VERIFIER_PRECHECK_G1G2.md` (same verifier session; lesson
bank applied: equal-R rule, grid headroom vs displaced MAPs, channel scoping, inert-null
preflight semantics, both-fire analysis, A-PF-1 rail precedence).

**Verdict: GO-WITH-AMENDMENTS — 4 BLOCKING (P7-1 … P7-4), 4 NON-BLOCKING (P7-5 … P7-8).**

---

## Part A — Checks that ran and PASSED (recomputed numbers)

### Recon-fact fidelity (check 3-class)

| drafted | artifact | verdict |
|---|---|---|
| iiib 2D mean_h 0.7842 (Δ +0.054), joint_r1 0.7966 (Δ +0.067), σ_h 0.0177/0.0216 | `run_20260804_postfix/*/combined_2d.json` — **recomputed with trapezoid (gradient) weights: 0.7842/0.7967, σ 0.0178/0.0217** | **exact under the trapezoid convention** (naive equal-weight gives 0.7803/0.7907 — see P7-2) |
| n = 1588; 41-point grid; 0.005 core / 0.01 tails | grids verified: 41 nodes on [0.60, 0.86], spacings {0.005, 0.01}; CSV = 65,108 rows = 1588 × 41 exactly | **exact** |
| harness 2D bias +0.008…+0.017, map_std 0.0032–0.0043 (n=1600) | `vdeep_1600_*` cells: 2D bias +0.0078…+0.0176, map_std 0.0031–0.0043 | **exact** (class) |
| Eddington-in-M −0.020 at `bayesian_statistics.py:5454` | comment block at 5448–5457 documents "2D-channel mean shifts −0.020 in h" with `G7row9_eddington_m_impact.json` ref | **verified** |
| σ_M mapping ln(10)·0.24 ≈ 0.55 | 0.5526 | **exact** (first-order caveat carried, §6.3) |
| event 889 present, iiib | in CSV ✓ | present; the 85× slope-dominance figure was **NOT independently recomputed** (assembly-dependent — covered by P7-2's registered convention + N-0 gate) |

### Instrument stream-alignment for the one-master-seed σ-grid (coordinator item)

Verified in code, both knobs:

- `sigma_z`: `z_obs = z_true + rng.normal(0.0, sigma_z, n)` (`pp_coverage.py:1323`) —
  scale-only, stream consumption invariant;
- `sigma_m_gal_frac`: `mass_obs = clip(mass_true[cat]·(1 + rng.normal(0.0,
  sigma_m_gal_frac, n_cat)), 1e-3, None)` (lines 1337–1342) — scale-only; masses are drawn
  **after** the redshift/direction/photo-z draws by documented design ("so the mass-free
  stream is unchanged", line 1330–1332), so draw order and counts are σ-independent;
- event-side draws (`eps` multivariate normal) use `sigma_dl_frac`/`sigma_mz_frac`, which are
  NOT swept; `p_draw`/host sampling independent of both swept knobs; clips are deterministic
  transforms.

⇒ **one master seed 20280411 across all T1/T2 cells yields latent-aligned, paired cross-cell
reads — valid.** V-prod cells correctly get their own seed (20280511: different d50/z_support
changes `p_draw`, so cross-venue pairing is not claimed — right call).

### Seeds, budget, headroom, quadrature

- **Seed freshness:** 20280411/20280511/20280611 fall in the verified free gap between the
  prodcal family (≤ 20271333) and the calibration-gate block (≥ 20280808; full inventory
  extracted in Part I §3), and are distinct from G-1/G-2's 20280311/20280399. Collision-free.
- **Cost arithmetic:** 181 nodes = (0.92−0.56)/0.002 + 1 ✓; ×1.38 = 181/131 ✓;
  12,918 s × 1.38 = 17,827 ≈ 17,850 ✓ ≈ 5.0 CPU-h. Wall ≈ 5–6 h at 15 workers vs 14 h
  request = ≥ 2.3× margin against the contended anchor ✓. Ceiling 160 CPU-h holds with large
  margin under every reading (but the "≈ 82 total" line double-counts — P7-6).
- **Grid headroom (Part-IV lesson):** top: worst registered expectation (H-T1b ×4 ⇒ 2D bias
  ≈ +0.045) at h_true 0.84 → center 0.885 + 3·map_std(0.0043) ≈ 0.90 < 0.92 ✓ (rail gate
  backstops); bottom: 1D fused −0.032 at 0.62 → 0.588, per-trial std ~0.005 at n=1600 ⇒
  P(rail at 0.56) ≈ 0 ✓. The deliberate 1D floor-rail at bad rungs is registered as an
  expected outcome with A-PF-1 precedence ✓ — the Part-II/IV lessons are correctly encoded.
- **n_z_quad = 160 at σ_z = 0.002:** valid transfer of the G-2 certification — the z-windows
  depend only on `h_grid.min()/max()`, and this campaign uses the identical wide bounds
  [0.56, 0.92] the G-2 cells ran; h_step does not enter the windows. ✓
- **Equal-R rule** respected (§3b: probe-scale byte/N-A comparisons declared VOID; N-1 at
  full R) ✓. No inert-null preflight trap: §3b demands engagement/rails/finiteness only, not
  probe pair non-degeneracy ✓.

### Both-fire sweep on the drafted bands

- H-T0a: FRAGILE ≤ 0.027 / ROBUST ≥ 0.0405 / MIXED between / OVERSHOOT < −0.01 — disjoint,
  two-sided ✓ (edges recomputed: ½·0.054 = 0.027, ¾·0.054 = 0.0405).
- H-T0b: interpretation bands on z_v are a partition (≤2 / 2–4 / ≥4) ✓.
- H-T1a: **PASS and FAIL can both fire if the anchor bias is small** (< 0.004): PASS cap
  max(0.002, 2·SE) vs FAIL ½·anchor — the G2-1 overlap class again → P7-3.
- H-L1/H-L2: no internal both-fire, but the 1D leg's meaning is unsupported and the 2D bias
  leg is overtight → P7-1; H-L2's "both channels calibrate" lacks a 1D-calibrates definition
  → P7-1(c).
- N-3 quantization guard: sound; its interaction with the good-corner deliverable → P7-7.

---

## Part B — The three coordinator questions

### (i) Closure-budget cross-instrument discipline — NOT yet well-defined (P7-4)

B-OWNED-SCATTER is clean (Δ_v, σ_boot both production-native). **B-OWNED-BUDGET is
ill-defined as drafted:** it asks whether production-native legs "plus the CLASS-level
harness attributions account for |Δ_v| within 2·σ_total" — but §6.1 (correctly) bars harness
magnitudes from production scale, and a class-level sign/collapse factor cannot "account for"
any part of a magnitude. As written, a reader cannot compute the branch condition, and a
hostile reader could smuggle harness magnitudes into the accounting. The budget equation must
be written out with production-native quantities only; harness legs certify the *class* of
the residual, never reduce it. → **P7-4 (BLOCKING)**.

### (ii) T0 statistical validity — resampling sound; the grid convention is verdict-relevant (P7-2)

Bootstrap over iid event columns for event-draw variance: valid, with the fixed-catalogue
limitation already registered (§6.2) ✓. Jackknife/leave-one-out: standard ✓. **But the
41-grid is non-uniform (0.005 core, 0.01 tails — verified), and the mean over it is
convention-sensitive at materiality scale:** naive equal-weight mean_h = 0.7803/0.7907 vs
trapezoid 0.7842/0.7967 — a −0.004/−0.006 discrepancy, i.e. **≈ the §5 materiality yardstick
(0.006)**. The quoted anchors match the trapezoid convention. Two further conventions are
load-bearing and currently unpinned: the per-event likelihood **column assembly** from the
16-column CSV (which columns form the 2D per-event likelihood), and the **physics-floor
zero-handling** (`combined_2d.json`: strategy "physics-floor", n_events_empty = 2) — the
bootstrap must apply the same floor or full-sample reproduction fails. One registered gate
covers all three at once: the scorer must reproduce the M3 anchors before any resampled
statistic is quoted. → **P7-2 (BLOCKING)**.

Additional finding: **joint_r1's posterior is right-edge truncated on the production grid**
(relative posterior at the top nodes 0.294/0.139/0.060/0.018 of peak; iiib 0.048→0.002).
Upward bootstrap resamples truncate at 0.86 ⇒ σ_boot biased LOW ⇒ z_joint biased HIGH —
verdict-relevant near the z = 2 edge (it pushes away from B-OWNED-SCATTER). Direction is
analyzable: B-OWNED-SCATTER firing despite truncation is conservative; near-threshold
readings are not. → **P7-5 (NON-BLOCKING disclosure + flag)**.

### (iii) Does the V-deep 2D +0.01 class contaminate H-L1's "2D constrains" leg?

**No off-twin correction is needed — but not for the drafted reason.** The venue 2D
noise-coupling bias (present identically in `off`, +0.0078…+0.0176 verified) is the
*measurand* of the landscape, not a nuisance: it is exactly the (σ_z × σ_m)-driven class the
grid varies, and its σ-independent component is already captured by the registered H-T1a/H-L2
FAIL branches (NEW-CLAIM intake). The selection lever's 2D contribution is certified
near-inert (−0.0006 measured; row #124 +0.001-class), and the anchor off twin polices it.
**However the 2D bias leg |2D bias| ≤ 2·SE (≈ 6.6e-4 at n=1600) is overtight for its
meaning:** a rung with a +0.001 bias — immaterial by §5 (0.006) and inside every calibration
band this project has ever registered — would be scored "2D fails to constrain", falsely
denying the frontier at that rung. Harmonize to max(0.002, 2·SE) (the H-T1a PASS edge).
→ folded into **P7-1(b) (BLOCKING)**.

**The genuine contamination is in the 1D leg, not the 2D leg** — see P7-1.

---

## Part C — The decisive finding (P7-1)

**H-L1's "1D starves" leg, as drafted, can only fire through the asymmetric-insertion
artifact — its meaning has no supporting arm (A8-v2 (b)).** The measured record this
campaign itself cites:

- V-deep **fused** 1D at σ_z = 0.035, n=1600: bias −0.032, cov68 0.000 — the rows #120–#124
  displacement, mechanism-owned as a **venue-regime property of the asymmetric [P2]
  insertion**, explicitly venue-scoped and NOT production-transferable;
- V-deep **off** 1D: bias −0.0013…−0.0017, cov68 0.667–0.758 — **calibrated**; and the
  G-2 grid shows the off-1D stays calibrated down to σ_z = 0.002 (bias −0.004-class, cov68
  0.65–0.89).

So at this venue the 1D channel **never genuinely starves in off form at any registered
rung**; the drafted leg (rail ≥ 0.5 OR cov68 ≤ 0.2, scored on fused cells) would fire "1D
starves" at GLADE quality **because of the insertion artifact**, and the "huge news" branch
(1D starves AND 2D constrains) could be assembled from a harness artifact the record already
owns as venue-scoped. The production-side 1D starvation the author means (photo-z railing,
the H0→0.86 class) is a **production phenomenon that T0 can measure natively** (per-channel
posteriors from `event_likelihoods.csv`) — that is where the "1D starves while 2D constrains"
statement must live.

---

## Part D — Amendments (exact quotable text)

**P7-1 [BLOCKING — A8-v2 (b); checks (iii) + arm-voidness].** Replace the H-L1 block with:

> - **H-L1-prod (the headline "1D starves / 2D constrains" read — production-native, T0).**
>   Per venue, from the registered T0 scorer: 1D-starves = the production 1D-channel
>   posterior is uninformative on the grid (68% HPD width ≥ ½ the grid span OR posterior
>   mode on the grid edge); 2D-constrains = the 2D posterior width σ_h with the H-T0b
>   closure z-score caveat attached (a 2D "constraint" is quoted as σ_h ⊕ σ_boot and carries
>   the B-branch outcome). This is the only arm on which the author's "1D starves while 2D
>   constrains" sentence may be quoted.
> - **H-L1-harness (the landscape frontier — class-level, venue-scoped).** Per grid rung:
>   2D-constrains = |2D bias| ≤ max(0.002, 2·SE) AND 2D cov68 ∈ [0.594, 0.766] AND
>   RMS ≤ 0.02, scored on the fused cells (the selection lever is certified +0.001-class);
>   1D reference = the **off-basis** 1D read at that σ_z (three off cells, §3-amended below;
>   the 1D channel is mass-blind, so three σ_z rungs cover the full grid): 1D-calibrated /
>   degraded / starved by the same band family. The fused-1D values are reported alongside
>   with the insertion delta (fused − off) explicitly attributed to the venue-scoped
>   asymmetric-insertion class (rows #120–#124, G-2 σ_z-collapse) — the fused-1D failure at
>   this venue is NEVER quoted as "1D starves".

And append to §3 (T1+T2 cells):

> - **1D off-basis cells (3):** selection_cell = off at (σ_z, σ_m_gal) = (0.035, 0.55),
>   (0.010, 0.55), (0.002, 0.55), same seed 20280411 — the 1D-leg basis for H-L1-harness
>   and H-L2 (≈ +3.7 CPU-h; total cluster cells 18).

And in H-L2, old text: "There exists a grid rung where BOTH channels calibrate" → new text:
"There exists a grid rung where the 2D channel calibrates (H-L1-harness legs) AND the
off-basis 1D read at that rung's σ_z is calibrated (|bias| ≤ max(0.002, 2·SE), cov68 ∈
[0.594, 0.766], rail ≤ 0.10)".

**P7-2 [BLOCKING — check (ii), convention pinning + continuity gate].** Append to §3 T0:

> **Registered T0 conventions:** (a) all grid moments use trapezoid weights
> w_i = gradient(h) on the non-uniform 41-grid (the naive equal-weight mean differs by
> −0.004/−0.006 — materiality-scale); (b) the per-event 2D likelihood is assembled from the
> CSV by the registered formula in the scorer header (pinned at freeze; the canonical
> Σ log L reference form of `posterior_combination.py`); (c) the physics-floor zero-handling
> of the production combine (strategy of record, n_events_empty = 2) is applied identically.
> **N-0 continuity gate (scored):** the T0 full-sample mean_h must reproduce the M3 anchors
> 0.7842 (iiib) / 0.7966 (joint_r1) to within 5e-4 per venue BEFORE any bootstrap/jackknife
> statistic is quoted; failure ⇒ STOP (convention mismatch), no T0 number enters the budget.

**P7-3 [BLOCKING — both-fire (G2-1 class)].** In H-T1a, old text:

> PASS(class-owned) if |bias(both-small)| ≤ max(0.002, 2·SE) AND each single toggle reduces
> |bias| by ≥ 25%; FAIL if bias(both-small) ≥ ½·bias(anchor) (a σ-independent 2D bias
> component exists — NEW CLAIM intake); MIXED else.

new text:

> **Engagement precondition:** |bias(anchor)| ≥ max(0.004, 5·SE) at the primary truth
> (else the collapse read is UNDETERMINED-BY-DESIGN, unscored — there is no amplitude to
> attribute). Under the precondition: PASS(class-owned) if |bias(both-small)| ≤
> max(0.002, 2·SE) AND each single toggle reduces |bias| by ≥ 25%; FAIL if
> |bias(both-small)| ≥ ½·|bias(anchor)| with the anchor's sign (a σ-independent 2D bias
> component exists — NEW CLAIM intake); MIXED else. The precondition floor 0.004 keeps the
> PASS cap (0.002) and FAIL edge (≥ ½·anchor ≥ 0.002) disjoint — no both-fire.

**P7-4 [BLOCKING — check (i), budget arithmetic definition].** Replace the B-OWNED-BUDGET
bullet with:

> - **B-OWNED-BUDGET:** z_v > 2 but the production-native residual
>   r_v = Δ_v − s_Edd (s_Edd = the documented −0.020 Eddington-in-M shift where its
>   configuration applies to venue v, else 0) satisfies |r_v| ≤ 2·σ_total,v with
>   σ_total,v = σ_boot,v ⊕ u(s_Edd) (u(s_Edd) registered at freeze; if no uncertainty is
>   documented it enters as a point value with that stated) ⇒ closure by budget; the
>   residual is quoted with its uncertainty. **Only production-native magnitudes enter this
>   arithmetic.** The T1 class attributions may only be cited as sign/collapse consistency
>   for the residual's CLASS (photo-z + mass-obs coupling), never as magnitude legs; the
>   fragility read (jackknife-889/top-k) modifies the INTERPRETATION (which Δ_v is quoted:
>   full vs without-889), not σ_total.

**P7-5 [NON-BLOCKING — T0 grid truncation].** Append to H-T0b:

> Grid-truncation diagnostic (registered): joint_r1's full-sample posterior retains ≈ 29% of
> peak at the penultimate node and 1.8% at h = 0.86 (iiib: 4.8%/0.2%) — upward resamples
> truncate, biasing σ_boot low and z_v high. The scorer reports the top-2-node relative mass
> per resample; if the median exceeds 0.05 for a venue, a z_v in (2, 4) is flagged
> UNDETERMINED-BY-GRID-TRUNCATION (B-OWNED-SCATTER firing anyway is conservative and
> stands).

**P7-6 [NON-BLOCKING — budget arithmetic clarity].** §2 old text: "15 cluster cells ⇒
≈ 75 CPU-h; +off twin and V-prod pair ≈ 82 CPU-h total" → new text: "15 cluster cells
(12 T2 grid + 1 anchor off twin + 2 V-prod) ≈ 70–75 CPU-h (off/V-prod cells are cheaper
than the fused V-deep class); +3 off-basis 1D cells (P7-1) ≈ +3.7 ⇒ ≈ 74–79 CPU-h total —
ceiling 160 CPU-h unchanged, wall at 18 workers still ≈ 5–6 h."

**P7-7 [NON-BLOCKING — good-corner quantization censoring].** Append to N-3:

> At good rungs the true map_std is expected below the 0.003 floor (G-2 rung-0.002 classes
> scale to ~0.0005 at n=1600): flagged cells quote σ_real and RMS as upper bounds
> ("< max(measured, 1.5·h_step)") in the landscape table — the deliverable's best-catalog
> rows are resolution-bounded, stated as such. A finer-grid (h_step 0.001) rerun of the ≤ 4
> best rungs (~+10 CPU-h) is a registered author option at readout, not run by default.

**P7-8 [NON-BLOCKING — venue correlation disclosure].** Append to §4:

> The two venues share the single event realization and CRB set (seed61000 recon): z_iiib
> and z_joint are strongly correlated draws of the same universe — venue agreement is ONE
> realization's evidence, never counted as two independent confirmations in the budget
> presentation.

---

*Part-VII verifier of record: same session as Parts I–VI. BLOCKING P7-1…P7-4 must land in
the freeze commit before local T0 scoring or cluster submission; the preflight (§3b) and
cluster-skill gates are unaffected and remain mandatory.*
