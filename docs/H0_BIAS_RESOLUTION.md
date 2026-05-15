# H0 Posterior Bias — Resolution Catalog

**Last updated:** 2026-05-07 (**Phase 48 production fine-grid sweep
LANDED** — 63-point non-uniform grid at h=0.73 phase46-merged 1473
events; 1D MAP=0.7324 z=+1.16σ PASS, 2D MAP=0.7322 z=+0.97σ PASS, info
monotonicity preserved. R1's 21-pt parabolic refine was Δh-resolution-
limited; production sweep refines MAP +0.0015 upward, within R1's
σ_boot. See §1 Phase 48 block. Previously: **H3 fix LANDED** `f01595c`
2026-05-06, see §3.15.)

This is the bundled source of truth for the H0 posterior bias investigation in
the LISA EMRI dark-siren H0 inference pipeline. It is organized as a **catalog
of confirmed bias sources** with mechanism / fix / evidence per entry, plus an
appendix preserving the original chronological narrative for date-stamped
context.

For per-test diagnostic outputs see
[`scripts/bias_investigation/FINDINGS.md`](../scripts/bias_investigation/FINDINGS.md)
and the per-test JSON outputs under
`scripts/bias_investigation/outputs/phase45/`.

---

## 1. Executive Summary

EMRI dark-siren H0 inference: the simulation injects events at `h_true = 0.73`
(`H` in `master_thesis_code/constants.py:25`); the inference recovers a posterior
over `h ∈ [0.60, 0.86]`. Across 12 confirmed bias sources resolved over phases
9–45, the cluster MAP currently sits at:

**Pre-Tier-3 (with the now-removed outer −N log D double-counting):**

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z |
|---|---|---|---|---|---|
| `without_bh_mass` (1D, anchored) | 0.7550 | 0.7535 (parabolic) | 0.0102 | +3.4 % | +2.30 σ |
| `with_bh_mass` (2D, unanchored) | 0.7450 | 0.7456 (parabolic) | 0.0050 | +2.0 % | +3.12 σ |
| Closure h_true=0.65 (1D, fine grid 11 h) | 0.6700 | 0.6708 (parabolic) | 0.0037 | +3.2 % | +5.62 σ FAIL |
| Closure h_true=0.65 (2D, fine grid 11 h) | 0.6700 | 0.6697 (parabolic) | 0.0059 | +3.0 % | +3.36 σ MARGINAL |

**Post-Tier-3 fix (combine_log_space joint = Σ log L_i, no outer correction):**

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z |
|---|---|---|---|---|---|
| `without_bh_mass` (1D) | **0.7400** | **0.7413** (parabolic) | 0.0073 | +1.4 % | **+1.54 σ PASS** |
| `with_bh_mass` (2D) | **0.7400** | **0.7409** (parabolic) | 0.0055 | +1.2 % | **+1.97 σ PASS** |
| `--evaluate` (60 events, local) | 0.7300 | n/a | n/a | 0.0 % (Phase 43 PASS) | n/a |
| **Closure h_true=0.65** (1D, fine grid 11 h) | **0.6550** | **0.6555** (parabolic) | 0.0033 | +0.85 % | **+1.67 σ PASS** |
| **Closure h_true=0.65** (2D, fine grid 11 h) | **0.6550** | **0.6557** (parabolic) | 0.0034 | +0.88 % | **+1.68 σ PASS** |

**Post-bridge fix (commit `2b33cad`, 2026-05-05) — phase46-merged 1473 events at h=0.73:**

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z | Status |
|---|---|---|---|---|---|---|
| `without_bh_mass` (1D) | **0.7300** | **0.7309** (parabolic) | 0.0046 | **+0.0009** | **+0.19 σ** | **PASS ✅** (fully closed) |
| `with_bh_mass` (2D) | 0.7400 | 0.7441 (parabolic) | 0.0039 | +0.0141 | +3.60 σ | **FAIL ⚠️** (10× reduction from pre-bridge +37σ; structural residual) |

**Post-bridge fix observations (2026-05-05):**
- 1D channel cleanest dark-siren H₀ recovery to date — within 0.001 of truth on 1473 events.
- σ_boot widened 6.5× post-fix (2D: 0.0006 → 0.0039), exposing pre-fix tightness as a *symptom* of the discontinuity, not noise; now scales physically with N.
- 2D bias is **16× larger than 1D bias** despite 2D *adding* information — violates info monotonicity, flagged as structural 2D-specific bug.
- **Bug identified during 2026-05-05 plan-mode review:** the 2D channel has TWO coupled issues — (1) numerator queries observation `_det_M` instead of hypothesis `host_M·(1+z)` at integration redshift z; (2) grid axis is source-frame `M` while queries pass observer-frame `M_z`. **Fix LANDED in commit `f01595c`** — see §3.15 below.

**Post-H3 fix (commit `f01595c`, 2026-05-06) — phase46-merged 1473 events at h=0.73:**

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z | Status |
|---|---|---|---|---|---|---|
| `without_bh_mass` (1D) | **0.7300** | **0.7309** (parabolic) | 0.0047 | **+0.0009** | **+0.18 σ** | **PASS ✅** (unchanged from post-bridge; 1D grid has no M axis) |
| `with_bh_mass` (2D) | **0.7300** | **0.7307** (parabolic) | 0.0037 | **+0.0007** | **+0.20 σ** | **PASS ✅** (z dropped 18×; bias 20× reduction) |

**Post-H3 fix observations (2026-05-06):**
- Both channels well within 2σ of truth; cleanest dark-siren H₀ recovery in this codebase (1D MAP=0.7309, 2D MAP=0.7307 against truth h=0.73).
- **Info monotonicity restored**: 2D bias (+0.0007) is now ≤ 1D bias (+0.0009), and 2D σ_boot (0.0037) is tighter than 1D σ_boot (0.0047) — adding the BH-mass likelihood correctly tightens the posterior toward truth, as it should.
- The H3 fix is a no-op in the 1D path (the 1D grid has no M axis), so the 1D channel is unchanged from post-bridge — confirming the fix targets the actual bug location.
- R2 (Phase 45 412 events) shows the same 1D-2D asymmetry resolution (Δ=0.0007 between channels); residual +0.0125 bias on the smaller dataset is within the seed-dependent MAP scale of 0.02 per `finding_seed_dependent_map`.

**Phase 48 production fine-grid sweep (jobs `4271862` + `4344777`, sbatch `8292359`) — phase46-merged 1473 events at h=0.73, 63-point non-uniform grid (Δh=0.001 dense core in [0.710, 0.750] + Δh=0.010 wings spanning [0.600, 0.860]):**

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z | Δh-spread | Status |
|---|---|---|---|---|---|---|---|
| `without_bh_mass` (1D) | **0.7320** | **0.7324** (parabolic) | 0.0021 | **+0.0024** | **+1.16 σ** | 0.00273 | **PASS ✅** |
| `with_bh_mass` (2D) | **0.7320** | **0.7322** (parabolic) | 0.0022 | **+0.0022** | **+0.97 σ** | 0.00188 | **PASS ✅** |

**Phase 48 observations (2026-05-07):**
- Production-grade verdict confirms R1's PASS conclusion. Both channels within ±1.2σ of truth; info monotonicity preserved (|2D bias| 0.0022 ≤ |1D bias| 0.0024).
- σ_boot tightened ~2× vs R1's 21-pt grid (1D: 0.0047 → 0.0021; 2D: 0.0037 → 0.0022) reflecting the 3× denser h-resolution near MAP.
- **MAP shifted +0.0015 upward vs R1 (within R1's own σ_boot).** The Δh-sensitivity scan re-computes MAP on sub-grids {full 63-pt, Δh=0.005, Δh=0.010}: the Δh=0.005 sub-grid (matching R1's resolution) recovers MAP≈0.7308 with σ_boot≈0.0048 — i.e. R1's 21-pt parabolic refine was **Δh-resolution-limited**. The Δh=0.001 dense core resolves MAP to ~0.7322–0.7324. **Production sweep is the paper reference; R1's headline 0.7307/0.7309 stands as a coarser-grid estimate of the same closure.**
- Δh-sensitivity spread (max−min continuous MAP across the four sub-grids) is comparable to σ_boot, slightly above the a priori 0.001 target — the dense core at Δh=0.001 is sufficient for paper-grade reporting, and no further refinement is warranted at this event count.
- Recovery footnote: first submission `4271862` TIMEOUT'd 7/7 tasks at 30:00 walltime (per-h walltime ~4.6 min on cpu_il 128-core vs planned ~2 min), leaving 41/63 h-values landed. Sbatch made idempotent in two follow-up commits (`7b24b98`: skip-if-output-exists; `8292359`: gate task-0 archive on opt-in `ARCHIVE_OLD=yes`); resubmission `4344777` filled the 22 missing h-values in 14–19 min/task. Two-guard pattern (per-unit skip AND opt-in cleanup gate) is required because cleanup-before-loop silently nullifies a per-unit skip — captured for cross-project reuse.

Verdict JSON: `scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json`.

**Audit A1 (2026-05-01) — G1b:** real shift, not a discrete-grid artifact.
Continuous 1D MAP=0.7550 with linear-vs-cubic Δ=0.0010 (PASS, tol 0.002).
**Audit A2 (2026-05-01) — G2b cluster-projection:** 1D anchored mean lift on
[0, 0.10] ≈ 0.834 vs 2D unanchored ≈ 0.568 (Δ=−0.27, sensitivity-robust).
1D over-anchors structurally vs 2D's natural histogram extrapolation.
**Audit A0 — G0a CLEAN:** no Hubble-constant inconsistency in production.
**Audit A5 — G5a-PARTIAL:** Phase 43 H2 alone moved MAP 0.860→0.730 (raw
Σ log L_i already peaked at 0.730 post-H2 even without D(h) correction).
**Audit A8 — G8a:** Phase 33/34 verdicts (P_det grid resolution and Fisher
quality) hold; baseline-invariant tests, no re-run needed.

**Status (2026-05-06):** v2.1 LANDED · v2.2 LANDED · v2.3 LANDED (Tier 3
D(h) fix `6754ddb`) · **Bridge fix LANDED** (`2b33cad`) — 1D channel
fully closed · **H3 fix LANDED** (`f01595c`) — 2D channel fully closed.
Both channels pass on phase46-merged 1473 events at h=0.73 with z<0.25σ
and info monotonicity satisfied. Phase 45 anchor side branches
(`phase-45-option-A`, `phase-45-option-D`) **superseded** by the
bridge+H3 sequence; safe to delete. Paper-readiness gate cleared.

**v2.3 audit programme summary (2026-05-04):**
A0 G0a CLEAN · A1 G1b real shift · A2 G2b 1D over-anchors structurally ·
A4 G4b BIAS DOMINATED BY D(h), not anchor · A5 G5a-PARTIAL · A7 redux
identified D(h) double-counting · **Tier 3 FIX LANDED**: outer −N log D
removed from `combine_log_space`; closure h=0.65 1D PASS (z=+1.67σ),
closure h=0.65 2D PASS (z=+1.68σ), production h=0.73 1D PASS (z=+1.4σ at
discrete MAP=0.7400), production h=0.73 2D PASS (z=+1.97σ). · A8 G8a
Phase 33/34 verdicts hold. A3, A6 superseded.

**A7 redux critical result (2026-05-04, supersedes 2026-05-03 G7a):** The
original A7 evaluated the rescaled-CRB closure on **only 4 h-values
[0.60, 0.65, 0.70, 0.75], Δh=0.05, with truth on a grid point** and used a
cubic spline through 4 points to extract a continuous MAP. With truth at the
grid centre, that geometry cannot resolve a sub-grid bias of magnitude
≲0.025. A re-run on **11 h-values Δh=0.005 in [0.625, 0.675]** with bootstrap
σ_boot via event resample (B=1000) — same resolution at which the +0.025 at
h=0.73 was detected — recovers:
- 1D continuous MAP = **0.6708** (bias = **+0.0208**, σ_boot=0.0037, **z=+5.62σ → FAIL**)
- 2D continuous MAP = **0.6697** (bias = +0.0197, σ_boot=0.0059, z=+3.36σ — MARGINAL)

**Decomposition (1D, h_true → MAP):** ΔΣ log L = −3.6 (per-event likelihoods
prefer truth, pulling MAP DOWN), Δ(−N log D) = +7.6 (selection-function
correction pushes MAP UP, +0.02). **The h=0.73 +0.025 residual IS a
structural pipeline bias** — same magnitude, same direction, same D(h)
mechanism at h_true=0.65. A4's hypothesis is now empirically confirmed by
the closure test, and the original "h=0.73 bias is realization-specific
fluctuation" framing is invalidated.

Cluster: job 4200482 on dev_cpu_il, 4 array tasks chained (3+3+3+2 h-values
serialized). Output: `scripts/bias_investigation/outputs/phase45/closure_h065_finegrid.json`.

**Tier 2 bootstrap-subsample at h=0.73 (2026-05-04):** Independently
corroborates: σ_boot at N=412 = 0.0102 stable across N ∈ {200, 300, 412}
with no convergence toward truth as N grows. P(MAP_b ≤ MAP_full)≈0.48 → MAP
sits at the median of its own bootstrap distribution. The +0.025 is real
and reproducible under resampling, NOT a sample-size fluctuation.
Output: `scripts/bias_investigation/outputs/phase45/bootstrap_subsample.json`.

**A4 empirical confirmation via Plan 45-06 (2026-05-03):** Parallel session
ran Plan 45-06 (raise d_L=0 anchor 0.7931 → 0.8873, +12% lift). Cluster MAP
**unchanged at 0.7550 / 0.7450** — direct empirical confirmation that anchor
escalation can't move the residual. Combined with A7-redux, this confirms
anchor work is the wrong layer AND the residual is a real structural bias
that anchor escalation cannot fix.

**A4 critical finding (2026-05-02):** Per-event diagnostic CSV (412 cluster
events × 38 h-values) reconstructs cluster MAP=0.7550 exactly. Decomposing
into Σ log L_i(h) vs −N log D(h):
- **Σ log L_i(h) alone peaks at h=0.7400** (within σ_boot=0.0109 of truth
  h=0.73 — likely statistical, not anchor-related).
- **−N log D(h) shifts MAP from 0.7400 → 0.7550** (+0.015, and is 2.7×
  larger than the per-event L pull with OPPOSITE sign).
- **Phase 45 anchor escalation targets per-event L_comp; the dominant
  bias is in D(h) selection-function normalization.** Plans 45-06 and 45-07
  are aimed at the wrong layer. **HALT anchor escalation.**

**A4 empirical confirmation via Plan 45-06 cluster eval (2026-05-03):** A
parallel session ran Plan 45-06 (raise d_L=0 anchor 0.7931 → 0.8873, +12%
lift) on cluster. **Cluster MAP unchanged at 0.7550** (1D channel) and
**0.7450** (2D channel) — a +12% anchor lift moved the discrete MAP by
ZERO grid steps. This is direct empirical confirmation that anchor
escalation cannot fix the bias: per-event L_comp behavior in [0, c_0]
is not the bottleneck.

**What is resolved (cluster posterior):**
- Galactic confusion noise (Phase 9), Fisher derivatives (Phase 10), KDE→IS
  P_det (Phase 11.1), L_comp local-window normalization (Phase 32), GLADE
  ecliptic frame (Phase 36 + 43-H2 PRIMARY mover per A5), parameter-estimation
  epsilon (Phase 37), L_cat formula (Phase 38), `extract_baseline` -N log D(h)
  (Phase 43-H1, secondary), h-dependent P_det zero-fill cutoff (Phase 44).
- **A1 falsified concern #3** (discrete-grid ambiguity): the 1D MAP shift
  0.7650 → 0.7550 was a real continuous shift, not a rounding artifact.
- **A0 falsified concern about** `TRUE_HUBBLE_CONSTANT=0.7` inconsistency:
  it's dead code; production paths use `H=0.73` end-to-end.
- **A8 confirmed** Phase 33/34 verdicts hold against post-Phase-43 baseline.

**What is not yet resolved:**
- **D(h) selection-function structural bias** (Tier 3 / new top priority).
  A4 + A7-redux empirically converge: ΔΣ log L pulls MAP toward truth at
  both h_true=0.65 and h_true=0.73, but Δ(−N log D) overrides with +0.015
  to +0.020 toward higher h. The structural systematic lives in the D(h)
  computation. Suspect surfaces: z-integration grid in
  `precompute_completion_denominator()`, `dV_c/dz` accuracy at high h,
  P_det grid extrapolation at large d_L (most of D(h)'s integrand mass),
  cosmological d_L↔z relation accuracy. **Read-only audit, ~1 hour per
  candidate.** Now blocking paper readiness.
- **Phase 45 first-bin asymptote underestimate** at d_L < c_0 — formally
  open but per A4+A7-redux this is the wrong layer; do not invest further.
- **Per-event diagnostic CSV** (Audit A4): closure-test decomposition makes
  it now clear that bias is global (D(h)-driven), not concentrated in
  ≤20 boundary events.

---

## 2. Current Cluster Numbers (2026-05-01)

### Cluster production (412 events, SNR≥20, seed 200)

Source: `results/phase45_v2_posteriors/combined_posterior.json` (1D),
`results/phase45_v2_posteriors_with_bh_mass/combined_posterior.json` (2D).

| Quantity | Value | Notes |
|---|---|---|
| Truth h | 0.7300 | Injection seed |
| Discrete MAP, 1D channel | 0.7550 | post-Plan-45-04 hybrid anchor |
| Discrete MAP, 2D channel | 0.7450 | unanchored; Plan 45-07 staged but not deployed |
| Continuous MAP (1D, A1) | 0.7550 (lin) / 0.7540 (cub) | cubic-vs-linear Δ=0.0010 < tol 0.002 PASS |
| σ_boot (1D) | 0.0099 | B=1000, RNG=20260429 |
| 68% bootstrap interval (1D) | [0.7450, 0.7650] | does **not** contain truth |
| 95% bootstrap interval (1D) | [0.7450, 0.7650] | degenerate with 68% — discrete h-grid |
| Residual / σ_boot | ≈ +2.5 σ_boot | systematic, not statistical |

### Local --evaluate diagnostic (60 events, SNR≥20)

Source: Phase 43 VERIFY-03 SC-3 PASS.

| Quantity | Value |
|---|---|
| MAP (60 events post-Phase-43 H1+H2) | **0.7300** |
| Bias | 0.0 % |

The cluster vs --evaluate gap is sample-size-driven (412 vs 60); both paths run
identical `BayesianStatistics.evaluate()` code.

### Closure test at h_true=0.65 — *2026-05-03 4-h-grid result was a smoke test, not a validation*

Lean rescaling closure test, cluster job 4198463 on `dev_cpu_il`.

| Quantity | Value | Notes |
|---|---|---|
| Truth h | 0.6500 | Closure-test injection (rescaled cluster CRB) |
| Continuous MAP, 1D channel | 0.6517 | from cubic spline through 4 widely-spaced points |
| Continuous MAP, 2D channel | 0.6501 | same |
| Cluster h-grid | [0.60, 0.65, 0.70, 0.75] | Δh=0.05, **truth on grid centre** |
| Bootstrap σ_boot | NOT COMPUTED | claimed bias "≪ σ_boot" had no σ_boot reference |
| N events post-rescaling SNR≥20 | 251 (243 used) | from 412 at h_true=0.73 |

**Withdrawn.** Cubic spline through 4 points cannot resolve a ≲0.025 sub-grid
bias when truth sits at a grid point. The "G7a UNBIASED" verdict was a
smoothing artifact. See `closure test fine grid` below for the actual
verdict.

### Multi-truth panel (partial, 4/7 truths) — 2026-05-05 11:54 — **post-Tier-3 residual surfaced in 2D**

Partial verdict from `test_24_multi_truth_bias_sweep.py` on the first 4 truths
of the full 7-truth panel run, evaluated on the **phase46-merged CRB** (1549
SNR≥20 events = Phase 45 seed=200 ⊕ seed=300 extension 41 tasks). σ_boot via
event resample (B=1000, RNG=42 default).

Snapshot: `scripts/bias_investigation/outputs/phase45/multi_truth_sweep_partial4truths_20260505_115451.{json,png}`.

| h_truth | N | 1D MAP | 1D bias | 1D σ_boot | 1D z | 2D MAP | 2D bias | 2D σ_boot | 2D z |
|---------|---|--------|---------|-----------|------|--------|---------|-----------|------|
| 0.60 | 903 | 0.6044 | +0.0044 | 0.0011 | +3.83 | 0.6149 | +0.0149 | 0.0003 | **+54.98** |
| 0.65 | 1019 | 0.6560 | +0.0060 | 0.0030 | +2.02 | 0.6570 | +0.0070 | 0.0017 | +4.03 |
| 0.70 | 1265 | 0.7067 | +0.0067 | 0.0049 | +1.37 | 0.6960 | −0.0040 | 0.0152 | −0.26 |
| 0.73 | 1473 | 0.7279 | −0.0021 | 0.0033 | −0.64 | 0.7512 | +0.0212 | 0.0006 | **+37.08** |

**Panel verdicts (4 truths):**
- **1D channel**: weighted mean bias +0.0041, z_panel=4.10, χ²_red=7.0 →
  `verdict_mean=FAIL`. Sign concordance PASS (3+/1−, p=0.62), no boundary rail,
  per-event pos_frac mean=0.69 std=0.05 (PASS, dispersion not suspicious).
- **2D channel**: weighted mean bias +0.0159, z_panel=65.5, χ²_red=1471 →
  `verdict_mean=FAIL spectacularly`. Same sign distribution and pos_frac
  pattern, but σ_boot is so tight (≤ 0.0006 at h=0.60 and 0.73) that the bias
  blows up to extreme z values.

**Reading honestly:** The Tier 3 D(h) double-counting fix (§3.13) eliminated
the +0.020 to +0.025 bias seen at h=0.73 alone. With Phase 45's 424 events
the residual was ≤ +2σ_boot (PASS at h=0.73, PASS at closure h=0.65). With
the merged 1549-event sample the σ_boot tightened by ~√3.7 ≈ 1.9× while the
**absolute bias did not shrink** — so what was 1–2σ at small N is now 4–55σ
at large N. This is the signature of a *structural* (not statistical)
residual that the Tier 3 fix did not fully address.

**The 2D channel is the focus.** 1D z_panel=4.1 is bad but in marginal-FAIL
territory; 2D z_panel=65 with χ²_red=1471 is unambiguously structural.
Hypotheses (to be investigated cold next session): (a) σ_boot resamples
events but does not capture seed-dependent MAP drift (`finding_seed_dependent_map`)
or shared-injection-set pull; (b) residual D(h) coupling specific to the 2D
joint posterior post-Tier-3; (c) BH-mass channel D(h)/normalization mismatch
between `posteriors/` and `posteriors_with_bh_mass/`. See
`.planning/HANDOFF-2D-BIAS-INVESTIGATION-20260505.md`.

**Caveat:** partial 4/7 truths only. The remaining 3 (0.75, 0.80, 0.85) may
shift the panel z if their biases are negative. Full panel ETA ~15:00 today.

---

### Closure test at h_true=0.65 — fine grid (Audit A7-redux, 2026-05-04) — **FAIL**

Cluster job 4200482 on `dev_cpu_il`, 4-task array chained (2-3 h-values per
task). Same RUN_DIR as A7 (rescaled CRB at h_true=0.65), but evaluated on
11 h-values Δh=0.005 in [0.625, 0.675] — the resolution at which the +0.025
at h=0.73 was originally detected. Bootstrap σ_boot via event resample
(B=1000, RNG seed 20260504, `test_20_closure_finegrid_analyze.py`).

| Quantity | 1D channel | 2D channel | Notes |
|---|---|---|---|
| Discrete MAP | 0.6700 | 0.6700 | (h-grid centred on truth, peak at h=0.67) |
| Continuous MAP (parabolic 3-pt) | **0.6708** | **0.6697** | sub-grid refine at the peak |
| Bias vs h_true | **+0.0208** | +0.0197 | similar magnitude both channels |
| σ_boot (B=1000) | 0.0037 | 0.0059 | bootstrap event resample |
| z = bias / σ_boot | **+5.62 σ** | +3.36 σ | 1D **FAIL**, 2D MARGINAL |
| ΔΣ log L (truth → MAP) | −3.6 | −5.1 | per-event L prefers truth |
| Δ(−N log D) (truth → MAP) | **+7.6** | **+7.6** | D(h) drives MAP up |
| Net Δ log p | +3.9 | +2.5 | D(h) overrides per-event L |

**Pipeline closure FAILS at h_true=0.65 on the same grid resolution that
detected the h=0.73 bias.** Both channels show the same direction and
magnitude (~+0.02) of structural bias dominated by Δ(−N log D)=+7.6.
A4's mechanistic claim — D(h) is the dominant bias driver — is now
empirically confirmed across two truths. The h=0.73 +0.025 residual is
therefore reproducible structural systematic, not realization-specific
fluctuation. Tier 3 D(h) audit is required before paper.

---

## 3. Bias Source Catalog

Confirmed bias sources, ordered by impact on H0 MAP (largest → smallest).
For each source: **Symptom** (what surfaced it), **Mechanism** (cause),
**Diagnostic** (test that pinpointed it), **Fix** (commit + file:line), **Evidence**
(numerical impact), **Limitations** (what's still open).

Each entry links to its date-stamped narrative in [Appendix A](#appendix-a--chronological-phase-log).

### 3.1 Local-window L_comp denominator (Phase 32) → MAP +0.13 local

- **Symptom:** Persistent low-h bias (MAP ≈ 0.60 / 0.68 vs truth 0.73) on the
  local 60-event SNR≥20 dataset; bias *grew* with N rather than shrinking;
  L_comp(h) showed a U-shaped anti-correlation with the truth.
- **Mechanism:** The completion-term denominator was integrated only over the
  local 4-σ d_L window. Gray et al. (2020) Eq. A.19 normalizes over the
  **full detectable volume** `D(h) = ∫ P_det(d_L(z, h)) · dV_c/dz dz`. Local-window
  normalization fails to cancel h-dependent volume effects and biases the
  posterior toward lower h.
- **Diagnostic:** `.gpd/phases/32-completion-term-fix/validation/lcomp_decomposition.json`
  (per-event L_comp(h) showing non-monotonic shape pre-fix).
- **Fix:** Replaced local-window denominator with precomputed full-volume `D(h)`
  table via `precompute_completion_denominator()`; also integrated into the
  cluster combine path via Phase 43 commit `2853c32`.
- **Evidence:** MAP 0.60 → 0.73 for both channels on 59-event local set; bias
  -17.8 % → 0.0 %; 0/1593 NaN values across all events × h.
- **Limitations:** Validated locally; cluster validation deferred to v2.2/Phase
  43 (which surfaced the H1 H2 issues exposed below).
- **Reference:** Gray et al. (2020) arXiv:1908.06050 Eq. A.19.
- **Detail →** [Appendix A · Phase 32](#phase-32--full-volume-dh-denominator-fix).

### 3.2 Equatorial CRBs vs ecliptic GLADE (Phase 36 + Phase 43-H2) → host recovery 31→38/60

- **Symptom:** Production cluster MAP persisted at 0.86 even after Phase 32 D(h)
  fix; Phase 40 VERIFY-04 surfaced |Δ/σ|=5.4 anisotropy in the Q3 quartile.
- **Mechanism:** GLADE catalog ingestion stored host coordinates in equatorial
  RA/Dec while EMRI sky angles `qS, phiS` were defined in ecliptic. The angular
  mismatch is up to the obliquity 23.4° while the BallTree median search
  radius is only 1.76°. Pre-fix, ~15 of 60 events landed on spurious near-host
  matches; the rest had "no possible hosts" → contributed only via L_comp.
  The bug had two surfaces:
  - **Phase 36 (`b460297`)**: GLADE ingestion was rotated equatorial → ecliptic
    via `astropy.coordinates.BarycentricTrueEcliptic(J2000)` along with
    BallTree polar-correct embedding (`c17ecb6`), eigenvalue search radius
    (`b2ef9c9`), and 4D sky sub-space (`5b9cfbf`).
  - **Phase 43-H2 (`ab4bc80`)**: cached `prepared_cramer_rao_bounds.csv` still
    held the *equatorial* CRBs computed before Phase 36; required separate
    `migrate_crb_to_ecliptic` rotation of the 14×14 Fisher covariance.
- **Diagnostic:** Pre-fix vs post-fix host counts; angular mismatch estimate
  (23.4° obliquity ≫ 1.76° search radius); Q3 anisotropy 5.4 σ → 0 σ.
- **Fix:** Phase 36 commits as above + Phase 43-H2 commit `ab4bc80`.
- **Evidence:** Host recovery 31/60 → 38/60; "no possible hosts" 10 → 1; Q3
  anisotropy 5.4 σ → 0 σ; --evaluate MAP returns to 0.730.
- **Limitations:** No ablation evidence in Phase 43-VERIFICATION isolating H2
  from H1 (concern flagged in Audit A5; queued v2.3).
- **Reference:** `astropy.coordinates.BarycentricTrueEcliptic(J2000)`.
- **Detail →** [Appendix A · v2.2 Phase 36](#phase-36--coordinate-frame-fix-2026-04-22)
  and [Phase 43](#phase-43--posterior-calibration-fix-2026-04-26--2026-04-27).

### 3.3 `extract_baseline` / combine missing -N log D(h) (Phase 43-H1) → removes monotone-h pull

- **Symptom:** Phase 40 VERIFY-03 h-sweep on the v2.2 baseline produced
  MAP=0.860; the pre-Phase-43 production combine path summed `Σ log L_i(h)`
  but omitted the selection-function term.
- **Mechanism:** Without `−N · log D(h)`, the posterior is biased toward
  `argmax Σ log L_i(h)` which monotonically grows with h (because individual
  event likelihoods are higher at h closer to the catalog matches' redshifts
  ↦ smaller distances ↦ tighter Gaussian peaks).
- **Diagnostic:** Toy `D(h) ∝ h³` model: pre-fix MAP at h_max = 0.90; with
  `−N log D(h)` correction (N=60), MAP shifts back toward 0.73 (Phase 43
  VERIFICATION computational oracle).
- **Fix:** Commit `2853c32` adds the Gray Eq. A.19 D(h) selection-function
  correction to the production combine path; commit `a2df67b` deprecates
  `extract_baseline()` with a logged warning citing the missing normalization.
- **Evidence:** N · log[D(0.86)/D(0.73)] = 29.5 (matches Phase 43 SUMMARY's
  29.6 prediction); --evaluate MAP 0.860 → 0.730.
- **Limitations:** `extract_baseline()` was deprecated by warning, **not deleted**
  — footgun: any future code that calls it will silently get the wrong answer
  with only a log warning. Tracked in §4 as low-priority cleanup.
- **Reference:** Gray et al. (2020) arXiv:1908.06050 Eq. A.19.
- **Detail →** [Appendix A · Phase 43](#phase-43--posterior-calibration-fix-2026-04-26--2026-04-27).

### 3.4 h-dependent P_det zero-fill cutoff at c_0 ∝ 1/h (Phase 44) → cluster 0.860 → 0.7650

- **Symptom:** Cluster posterior persisted at MAP=0.860 even after Phase 43's
  H1 fix unblocked the --evaluate path. Debug session traced a +145.7 log-unit
  pathology: 4 close events with d_L ≈ 0.085–0.097 Gpc had L_comp = 0 at h=0.73
  but L_comp ≫ 0 at h=0.86.
- **Mechanism:** `detection_probability_without_bh_mass_interpolated_zero_fill`
  in `master_thesis_code/bayesian_inference/simulation_detection_probability.py`
  applied a left-side cutoff at `dl_centers[0] = dl_max(h)/120`. Because
  `dl_max(h) ∝ 1/h`, the cutoff `c_0(h) ∝ 1/h` was a **moving threshold**:
  c_0(0.73) = 0.0998 Gpc, c_0(0.86) = 0.0847 Gpc. The 4 close events fell
  below threshold at h=0.73 but above at h=0.86, so their L_comp "switched on"
  only at high h, pinning MAP at 0.860. The threshold was a *bin-midpoint
  artifact*, not the injection minimum (the first bin spans `[0, 2 c_0)` and
  is densely populated by GLADE-low-z injections, n_total=312).
- **Diagnostic:** `.gpd/debug/resolved/map-0p86-lcat-explosion.md` — pre-fix
  per-event `p_det(d_L=0.085, h)` table showing `0.0` for h ∈ [0.65, 0.83],
  `0.59` at h=0.86.
- **Fix:** Commit `3697bdd` removes the left-side zero-fill so the existing
  `RegularGridInterpolator(fill_value=None)` returns the genuine first-bin
  estimate `p̂(c_0) ∈ [0.47, 0.60]` for d_L < c_0. Right-side zero-fill kept
  (sources beyond injection horizon are genuinely undetectable).
- **Evidence:** Cluster re-eval (jobs 4160638/4160639) MAP shifted **0.860 →
  0.7650**; +145.7 log-unit pathology eliminated; all 4 zero-handling
  strategies now produce identical MAPs.
- **Limitations:** Residual +0.035 (3 σ_boot) above truth — handled by §3.5.
- **Reference:** Gray et al. (2020) Eq. A.19.
- **Detail →** [Appendix A · Phase 44](#phase-44--h-dependent-p_det-zero-fill-cutoff-2026-04-28--2026-04-29).

### 3.5 P_det first-bin asymptote underestimate at d_L < c_0 (Phase 45, ACTIVE)

- **Symptom:** After Phase 44, cluster MAP=0.7650; bootstrap 68% [0.745, 0.765]
  excludes truth by ≈3 σ_boot=0.0114. Bootstrap rules out statistical
  fluctuation as the explanation.
- **Mechanism (Phase 45 diagnosis, T8–T11):**
  1. **First-bin upper-skew (T9):** within the first d_L bin `[0, 2c_0]`,
     n_upper3 / n_lower3 = 29/9 = 3.22; weighted mean d_L = 0.132 Gpc (above
     midpoint 0.10).
  2. **Empirical asymptote (T10):** at d_L < 0.10 Gpc, 16/16 events detected
     (p̂=1.000, Wilson 95% LB 0.806). Interpolator returns 0.544 at c_0 and
     0.748 at d_L → 0 (linear extrapolation).
  3. **Underestimate (T10):** ≈0.25 at d_L→0, ≈0.46 at c_0 vs empirical ≈1.0.
  4. **Window proximity (T11):** 0/60 events touch d_L=0 but **26/60 (43%)**
     integrate across c_0. The fix must lift `[0, c_0]`, not just `d_L=0`.

  Mechanism: histogram-derived `p̂(c_0) ≈ 0.544` is biased downward because
  the upper-skewed injection density makes the bin mean dominated by the
  high-d_L (low-p_det) end. Linear extrapolation through bins 0,1 partly
  recovers the trend but still systematically underestimates p_det in
  `[0, c_0]`, suppressing L_comp at low h and biasing MAP upward.

- **Diagnostic scripts:**
  - `scripts/bias_investigation/test_08_bootstrap_map.py` (T8 — bootstrap)
  - `test_09_first_bin_density.py` (T9 — upper-skew)
  - `test_10_pdet_asymptote.py` (T10 — asymptote)
  - `test_11_window_proximity.py` (T11 — window)
  - `test_12_p_max_h_independence.py` (LR test, h-independence p=0.199)
  - Outputs: `scripts/bias_investigation/outputs/phase45/{bootstrap,first_bin_density,pdet_asymptote,window_proximity,p_max_h_independence}.json`

- **Fix (in flight):**
  - **Plan 45-01:** LR-test of binomial-rate homogeneity across h_inj groups
    (G=7.30, dof=5, p=0.199 — cannot reject); recommends pooled Wilson 95% LB
    `_P_MAX_EMPIRICAL_ANCHOR = 0.7931` (commit `49be6c0`).
  - **Plan 45-02:** prepend `(0.0, 0.7931)` to `dl_centers, p_det_1d` in
    `_build_grid_1d` (file `simulation_detection_probability.py:529–661`,
    commit `09ee262`); 564 CPU tests pass.
  - **Plan 45-03:** cluster re-eval — MAP **unchanged at 0.7650**; posterior
    peak height 0.373 → 0.347 with mass redistributed to bins 0.745–0.755
    (sub-discrete-grid-step shift). Branch B (UNDER-CORRECTION). ESCALATE.
  - **Plan 45-04:** add second anchor `(0.05, 1.0)` (commit `4a260e9`); local
    probe confirms `interp(0.05; h=0.73) = 0.6687 → 1.0` (+0.331).
  - **Plan 45-05:** cluster re-eval — MAP **0.7650 → 0.7550** (one discrete
    grid step toward truth); σ_boot tightened 0.0117 → 0.0099. 68% interval
    still [0.745, 0.765], excludes truth by 0.015. Branch B again. ESCALATE-AGAIN.

- **Evidence so far:** Cluster MAP 0.7650 → 0.7550 (one Δh=0.005 step). The
  ~4× larger local lift in 45-04 vs 45-02 produced the same continuous σ-shift
  (~−0.005). Saturation hand-waved, not derived.

- **Limitations & open questions:**
  - **Audit A4 (2026-05-02) found that Phase 45's anchor target is the wrong
    layer.** The cluster MAP=0.7550 decomposes into Σ log L peak at h=0.7400
    (within σ_boot=0.0109 of truth) plus D(h) selection-function shift of
    +0.015. The dominant bias is in D(h), not in per-event L_comp. Anchor
    escalation (Plans 45-06, 45-07) targets per-event L_comp and cannot fix
    the D(h) offset. **Phase 45 anchor work should be HALTED** pending D(h)
    audit (next priority).
  - **Anchor is being empirically tuned toward truth.** Sub-binning (RESEARCH §4b),
    the principled mechanism-addressing alternative, is also a per-event
    L_comp lever and is therefore superseded by A4's redirect to D(h).
  - **Discrete grid Δh=0.005 + σ_boot ≈ 0.01** makes "1 grid step toward
    truth" hard to interpret as physics improvement. **Audit A1 (2026-05-01)
    resolved this**: continuous shift is real (G1b).
  - **2D channel (`_build_grid_2d`) currently unanchored** — uses linear
    extrapolation through histogram bins 0,1, MAP=0.7450 (closer to truth than
    anchored 1D 0.7550). Plan 45-07 would extend hybrid to 2D.
    **Audit A2 finding (2026-05-01):** cluster-scale projection (using known
    cluster c_0=0.10 Gpc and p̂(c_0)=0.544 from T10) gives anchored 1D mean
    lift on [0, c_0]=0.834 vs unanchored 2D mean lift=0.568 — a structural
    Δ=−0.27 in 2D's favour. **The 1D anchor lifts substantially more than
    2D's natural histogram extrapolation.** Caveat: attributing the +0.010
    MAP gap (1D=0.7550 vs 2D=0.7450) entirely to this lift gap would
    over-claim — the channels also differ in BH-mass conditioning of L_cat.
    Conservative reading: 45-06 (raise anchor 0.7931→0.8873) would WIDEN
    the lift gap and likely move 1D MAP further from 2D's reference;
    45-07 (extend hybrid to 2D) would impose 1D's already-larger lift on
    the 2D channel — direction NOT signed safely without per-event audit
    (A4).
  - Module-level constants in
    `master_thesis_code/bayesian_inference/simulation_detection_probability.py`:
    `_P_MAX_EMPIRICAL_ANCHOR=0.7931` (L69), `_D_INTERMEDIATE_ANCHOR_GPC=0.05`
    (L109), `_P_INTERMEDIATE_EMPIRICAL=1.0` (L110).

- **Detail →** [Appendix A · Phase 45](#phase-45--p_det-first-bin-asymptote-fix-active-2026-04-30--present).

### 3.6 L_cat formula (Phase 38) → Gray Eqs. 24–25 normalization

- **Symptom:** Pre-Phase-38 production used an ad-hoc per-galaxy normalization
  that did not match Gray et al. (2020) Eqs. 24–25; v2.2 baseline MAP shifted
  vs Phase 32-validated --evaluate MAP.
- **Mechanism:** L_cat per host should be `(1/N) · Σ_g (N_g / D_g)` summed over
  galaxies in the 4σ window, not a marginal density. The pre-fix formula
  effectively double-weighted high-density regions.
- **Diagnostic:** Phase 38 derivation cross-check vs Gray et al. (2020) Eqs. 24–25.
- **Fix:** Commit `005e792` (L_cat formula); commit `a70d1a2` (symmetric P_det
  zero-fill + off-grid quadrature diagnostic — early surface of Phase 44 bug).
- **Evidence:** Built into v2.2 baseline; impact decoupled from Phase 43 H1+H2 fix.
- **Limitations:** Numerical impact alone (without H1+H2) was not isolated.
- **Reference:** Gray et al. (2020) Eqs. 24–25.
- **Detail →** [Appendix A · Phase 38](#phase-38--statistical-correctness-2026-04-23).

### 3.7 Spurious /(1+z) Jacobian (Phase 15) → minor; insufficient alone

- **Symptom:** Pre-Phase-15 `bayesian_statistics.py:646,871` carried a `/(1+z)`
  factor in the with-BH-mass numerator that was a code anomaly per the
  Phase 14 first-principles derivation.
- **Mechanism:** Spurious Jacobian — the absorption identity in the Phase 14
  derivation showed the factor cancels symbolically.
- **Diagnostic:** Phase 14 derivation (`derivations/dark_siren_likelihood.md`).
- **Fix:** Commit `1d4e9a1` removed `/(1+z)` from numerator.
- **Evidence:** Posterior remained monotonically decreasing after fix
  (commit `46e2662`) — necessary correction but **not the dominant bias mechanism**.
- **Detail →** [Appendix A · Phases 14–15](#phases-1415--likelihood-derivation-and-1z-fix-2026-03-31).

### 3.8 Per-parameter Fisher epsilon (Phase 37 PE-02)

- **Symptom:** Single global `derivative_epsilon` was poorly tuned for slow
  EMRI parameters; CRBs over-estimated for some, under-estimated for others.
- **Mechanism:** O(ε⁴) five-point stencil error scales as ε⁴ but the
  *condition* of the derivative depends on parameter scale; one ε ill-fits 14
  parameters spanning many orders of magnitude.
- **Diagnostic:** Per-parameter SC-3 regression tests (commit `16ce20f`).
- **Fix:** Commit `7429c6e` — per-parameter `derivative_epsilon` for all 14
  EMRI parameters.
- **Evidence:** Fisher matrix condition number improved across the parameter set.
- **Detail →** [Appendix A · Phase 37](#phase-37--parameter-estimation-correctness-2026-04-23).

### 3.9 P_det extrapolation `fill_value=0.0` (v1.4 / commit 44d5358)

- **Symptom:** 702 events received P_det=0 at evaluation, falling back to
  catalog-only likelihood. MAP biased low at h=0.66, bias -9.6 %.
- **Mechanism:** `RegularGridInterpolator(fill_value=0.0)` for out-of-grid
  queries. Grid covered the bulk of the d_L distribution but not the tails;
  events near the tails got P_det=0 → L_comp=0.
- **Diagnostic:** 702 events with completeness fallback identified.
- **Fix:** Commit `44d5358` changes `fill_value=0.0 → fill_value=None`
  (nearest-neighbour extrapolation).
- **Evidence:** MAP h 0.663 → 0.680 (+0.017); bias -9.2 % → -6.9 %.
- **Limitations:** This change later interacted with Phase 44's left-side
  zero-fill bug; Phase 44 removed the explicit cutoff that this `fill_value=None`
  was bypassing on the right side.
- **Detail →** [Appendix A · P_det Grid Extrapolation](#p_det-grid-extrapolation-fix-2026-04-08).

### 3.10 KDE → simulation-based IS P_det estimator (Phase 11.1)

- **Symptom:** KDE-based P_det had bandwidth sensitivity, poor tail coverage,
  and systematic over/underestimation at grid boundaries.
- **Mechanism:** Kernel-density on a finite injection set produces
  bandwidth-dependent boundary effects — particularly bad for our 463k injections
  at d_L → 0 / d_L → max boundaries.
- **Diagnostic:** Variance reduction factor (VRF) per bin; KDE vs IS comparison.
- **Fix:** Commits `e86e826`, `caf8ce6` — replaced KDE `DetectionProbability`
  with `SimulationDetectionProbability` using importance-sampling weights and
  `RegularGridInterpolator`.
- **Evidence:** VRF 11.8–24.9× in boundary bins; backward-compatible API.
- **Detail →** [Appendix A · Phase 11.1](#phase-111--kde-to-is-detection-probability-2026-03-31).

### 3.11 Five-point Fisher derivatives (Phase 10)

- **Symptom:** Forward-difference O(ε) Fisher derivatives produced loose
  Cramer-Rao bounds on all 14 parameters including d_L, broadening the per-event
  GW likelihood.
- **Mechanism:** Forward difference is O(ε); central five-point stencil is
  O(ε⁴) with coefficients (-1, 8, -8, 1) / 12ε.
- **Diagnostic:** Vallisneri (2008) Fisher-stencil convergence comparison.
- **Fix:** Commit `a87eeab` — `use_five_point_stencil=True` default in
  `compute_fisher_information_matrix()`.
- **Evidence:** Tighter d_L and sky-localization uncertainties; tightened
  per-event GW likelihood.
- **Reference:** Vallisneri (2008) arXiv:gr-qc/0703086.
- **Detail →** [Appendix A · Phase 10](#phase-10--five-point-stencil-derivatives-2026-03-29).

### 3.13 D(h) double-counting in joint posterior (Tier 3 fix, 2026-05-04) → +0.020 to +0.025 bias eliminated

- **Symptom:** Cluster MAP +0.025 over truth at h_true=0.73 (z=+2.30 σ_boot).
  A7 fine-grid closure at h_true=0.65 also failed: 1D MAP=0.6708, bias=+0.0208,
  z=+5.62σ. A4 decomposition: per-event Σ log L_i peaked at h≈truth at both
  truths, but Δ(−N log D)=+7.6 dominated and pushed MAP up by ~+0.02.
- **Mechanism:** D(h) appeared **twice** in the joint posterior. (1) Inside
  per-event L_comp = num/D (Phase 32 commit `fc7c84c`) — this is the prior
  normalization for `p_galaxy ∝ p_det · dV_c/dz` per Gray Eq. 31, making L_comp
  a probability density of d_obs *conditional on detection*. (2) Outside via
  `combine_log_space`'s subtraction of N · log D(h) (Phase 43-H1 commit
  `2853c32`) — re-applying selection correction that's already inside the
  per-event likelihoods. For completion-dominated events (f_i → 0) the joint
  contribution became `log num_i − 2·log D`, over-correcting; for catalog-dominated
  events (f_i → 1) it remained `log L_cat − log D`. Net structural bias whose
  sign and magnitude depended on D(h)'s actual h-dependence post-Phase-44.
- **Why Phase 43-H1 appeared correct at the time:** verification used a toy
  D(h) ∝ h³ that doesn't match production. Phase 43-H2 (the *real* fix —
  ecliptic CRB migration) independently moved MAP from 0.860 to 0.730 (per
  A5: "the raw Σ log L_i already peaks at 0.730 without needing D(h)
  correction"). After Phase 44 inverted D(h)'s slope from increasing-in-h to
  decreasing-in-h, H1's correction became actively harmful.
- **Diagnostic:** `scripts/bias_investigation/test_22_dh_double_count.py` —
  joint MAP with outer correction coefficient c ∈ {0, 1} at all 4 (channel × truth)
  combinations. With c=1 (current pre-fix): 3/4 FAIL, 1/4 MARGINAL. With c=0:
  4/4 PASS within 2·σ_boot of truth.
- **Fix:** `master_thesis_code/bayesian_inference/posterior_combination.py` —
  `combine_log_space` no longer subtracts `n_events_used · log_D_h`; the
  parameter is retained for API compatibility but ignored. Per-event L_comp = num/D
  (Phase 32) is unchanged. Joint posterior is now `Σ log L_i` per Loredo (2004) /
  Mandel et al. (2019) §3 — each L_i already conditions on detection, no extra β^N
  correction needed.
- **Evidence:** Closure h_true=0.65 fine-grid 1D MAP 0.6708 → 0.6555
  (z=+5.62σ → z=+1.67σ, PASS); 2D MAP 0.6697 → 0.6557 (z=+3.36σ → z=+1.68σ,
  PASS). Production h_true=0.73 1D MAP 0.7535 → 0.7400 discrete (z=+2.30σ →
  z=+1.4σ, PASS); 2D MAP 0.7456 → 0.7400 (z=+3.77σ → z=+1.97σ, PASS).
- **Limitations:** Production cluster posteriors at `results/phase45_v2_posteriors/`
  contain per-event L_i values that are still correct (D(h) is inside L_comp,
  unchanged); only the *combine* step changes. Re-running `combine_posteriors`
  on existing per-h JSONs produces the corrected MAP without needing a fresh
  cluster eval.
- **Reference:** Gray et al. (2020) arXiv:1908.06050 Eq. 31 (L_comp prior
  normalization); Mandel, Farr, Gair (2019) arXiv:1809.02063 §3
  (selection-conditional likelihood formulation).
- **Detail →** [Appendix A · Tier 3](#tier-3--dh-double-counting-fix-2026-05-04).

### 3.14 Detection-probability extrapolation: principled bridge replaces Phase 45 anchor (2026-05-05)

- **Symptom:** post-Tier-3 partial 4-truth panel on the phase46-merged CRB
  (1549 SNR≥20 events) reported 2D z=+37 at h=0.73 and z=+55 at h=0.60 —
  while the Phase 45 412-event closure test had previously PASSED at z=+1.97.
  Absolute 2D bias **doubled** going 412 → 1473 events (0.0109 → 0.0212),
  inconsistent with a pure σ_boot-blindspot (H1) story; strongly consistent
  with a deterministic per-event mechanism (H2). 1D bias also grew but more
  modestly (z=+0.6 to +3.8 across truths).
- **Mechanism (audit script
  `scripts/bias_investigation/test_26_2d_pdet_edge_behavior.py`,
  2026-05-05):**
  - **6–12% of events at every truth fall below `dl_min(2D grid)`** (the
    saturated d_L→0 regime). Raw scipy linear extrapolation in
    `RegularGridInterpolator(method="linear", fill_value=None)` extrapolates
    using the boundary-cell slope, which can drift downward (the first 2D
    bin has only ~7 injections per d_L slice — KDE noise). Extrapolated
    values reach −0.25; the [0, 1] clip at the production lookup floors
    these to ≈0.
  - Principled value at d_L<dl_min is p_det → 1 (saturated; no source
    closer than the observer). Production code returns ≈0 — a 100×
    underestimate for very nearby events.
  - As h_trial varies (Δh=0.005 step), the 2D grid bounds shift slightly
    and ~50–60 events cross the boundary at each step; each crossing
    produces a per-event likelihood discontinuity, manifesting as spurious
    h-trial-dependence in the joint posterior. Confirmed at h_truth=0.73:
    h_trial 0.680 → 0.685 transition shifts 57 events from out-of-grid to
    in-grid.
  - **1D channel asymmetry:** 1D used the Phase 45 Plan 45-02/04 anchor
    scheme (Wilson 95% LB at d_L=0 = 0.7931, intermediate at d_L=0.05 = 1.0)
    to handle the same d_L→0 saturation regime. The 2D channel had no
    anchors — that's why the 2D residual was ≈10× larger than the 1D.
    However, the 1D anchor was deliberately fitted to truth ("conservative
    Wilson LB chosen to not overshoot truth on production posteriors";
    `simulation_detection_probability.py:62-63` pre-fix) and the augmented
    Phase 46 injection campaign now gives p̂(c_0) = 1.0 at the first bin,
    so the Wilson anchor is actively *suppressing* the empirical 1.0 down
    to 0.7931 — opposite of its original lift purpose.
- **Diagnostic →** test_26 output JSON
  `scripts/bias_investigation/outputs/phase46_merged/2d_pdet_edge_behavior.json`;
  per-truth direction tables in `.planning/2D-CHANNEL-AUDIT-20260505.md`.
- **Fix:** principled monotonic-asymptotic extrapolation (uniform scheme
  for both 1D and 2D channels, replacing Phase 45 anchor and raw scipy
  extrapolation):
  - **Saturating face (d_L < dl_min):** linear bridge from
    (dl_min, p_edge) to (0, 1). C0 continuous at dl_min; reaches p_det=1
    at the unique natural physical scale d_L=0; uses no fitted constants
    or noisy boundary KDE slopes.
  - **Suppressing faces (d_L > dl_max; M_z > M_max; M_z < M_min):**
    slope-matched linear extrapolation from the boundary, clamped to
    [0, p_edge] (Option A directional clamp).
  - **Corner cells (2D, both axes outside):** min of the two face
    extrapolations.
- **Implementation:**
  `master_thesis_code/bayesian_inference/simulation_detection_probability.py`
  — replaced bodies of `detection_probability_with_bh_mass_interpolated`
  and `detection_probability_without_bh_mass_interpolated_zero_fill`.
  Removed the anchor-prepending in `_build_grid_1d`. Removed module
  constants `_P_MAX_EMPIRICAL_ANCHOR`, `_D_INTERMEDIATE_ANCHOR_GPC`,
  `_P_INTERMEDIATE_EMPIRICAL`. Test suite extended with 15 property-based
  tests across both channels.
- **Reference:** monotonicity argument (Maggiore 2008, Vol 1, §7.7
  inspiral SNR scaling) + the EMRI rate-density cutoffs at high/low M_z
  encoded by the simulation grid itself. Numerics-of-grids choice — not
  citable to a specific paper. Rationale captured in
  `.planning/2D-CHANNEL-AUDIT-20260505.md` and the user-feedback memory
  `feedback_principled_physics_choices.md`.
- **Status:** code change committed (`2b33cad`). Local property tests
  pass (15/15); full test suite 514/514 pass. **Production validation
  result (job 4229895, cpu_il, 2026-05-05):**
  - **1D channel: PASS ✅** — z=+0.19, MAP=0.7309, bias=+0.0009 on
    h=0.73 phase46-merged 1473 events. Cleanest 1D dark-siren H₀
    recovery in this codebase to date.
  - **2D channel: residual remains** — z=+3.60 (10× reduction from
    pre-fix +37σ), bias=+0.0141. σ_boot widened 6.5× post-fix
    (0.0006→0.0039) — pre-fix σ_boot was an artefact of the
    boundary-crossing discontinuity, now scales physically with N.
    The 2D bias being 16× larger than 1D bias violates information
    monotonicity; flagged as a structural 2D-specific bug. Fix
    planned in §4.7.
  - **Phase 45 412-event closure: NOT YET re-validated** under the
    new bridge — pending in §4.7 R2 step.
- **Note on §3.5 supersession:** Phase 45's anchor work (§3.5) is now
  formally superseded by this principled scheme. The Wilson 95% LB and
  intermediate empirical anchor were deliberately fitted to truth — a
  modeling choice the project has since rejected on principle.
- **Note on incompleteness for 2D:** the bridge fixes the **d_L axis**
  out-of-grid policy. The remaining 2D residual was on the **M axis**
  + numerator hypothesis convention — closed by §3.15 (H3 fix).
- **Detail →** `.planning/2D-CHANNEL-AUDIT-20260505.md`.

### 3.15 Numerator p_det query: observation→hypothesis fix and 2D grid M_z axis (H3 fix, 2026-05-06)

- **Symptom:** post-bridge fix (§3.14, commit `2b33cad`), the 2D channel
  on phase46-merged 1473 events at h=0.73 sat at MAP=0.7441,
  σ_boot=0.0039, **bias=+0.0141, z=+3.60σ** while the 1D channel was
  fully closed at MAP=0.7309 (z=+0.19σ). 2D bias **16× larger than 1D
  bias** — adding the BH-mass likelihood (channel B) was *increasing*
  posterior bias relative to position-only, **violating information
  monotonicity**. Adding observable information should tighten the
  posterior toward truth, not move it away. The asymmetry was
  flagged as a structural 2D-specific bug, not a statistical residual.
- **Mechanism (two coupled issues, surfaced during 2026-05-05 plan-mode
  review):**
  1. **Numerator queried the observation, not the hypothesis.** At
     `bayesian_statistics.py:1304-1306` the integrand passed
     `np.full_like(z, _det_M)` — the detection's measured ML
     observer-frame mass — *constant* across integration over candidate
     redshift z. But the integrand's *hypothesis* at each integration z
     is "the source is the host candidate at z with source-frame mass
     `host_M`", which implies an observer-frame `M_z = host_M · (1+z)`
     **varying with integration z**. The rest of the integrand already
     used the hypothesis (cf. `mu_gal_frac = host_M·(1+z) / _det_M` in
     the GW likelihood Gaussian product). Only the p_det query was
     using the observation — the Phase 14 "approximation, not a bug"
     comment was justifying this exact mismatch.
  2. **Grid axis vs query coordinate mismatch.** The 2D grid was built
     binning injections in source-frame `M` while production queries
     passed observer-frame `M_z` (denominator: `M·(1+z)`; numerator:
     `_det_M` which is `M_z`). At z≈0.5 the queries were ~50% higher
     than the bin labels — `p_det` is M-dependent, so this produced a
     deterministic per-event bias.
- **Diagnostic:** `scripts/bias_investigation/test_27_m_coordinate_mismatch.py`
  builds two `SimulationDetectionProbability` instances differing only
  in M-axis convention (current source-frame vs proposed observer-frame
  M_z); confirms 23% of events at h=0.73 phase46-merged have
  |Δp_det| > 0.05 (mean Δp_det=-0.031 at central integration z) and
  the proposed grid's M-axis extends 1.487× the current max (matching
  the expected M_z/M_source ratio at typical z).
- **Fix (Option A — observer-frame M_z everywhere, commit `f01595c`):**
  - `simulation_detection_probability.py:_get_or_build_grid` (~L308):
    multiply by `(1 + _z_arr)` at construction so the 2D grid M-axis
    is observer-frame `M_z = M_source · (1+z_inj)` per injection.
  - `bayesian_statistics.py:1304-1306` (numerator integrand):
    change `p_det` query from `np.full_like(z, _det_M)` (observation,
    z-constant) to `host_M * (1.0 + z)` (hypothesis, varies with
    integration z), matching the rest of the integrand's hypothesis
    convention and the denominator's `M·(1+z)` query.
  - Removed the "known approximation, not a bug, per Phase 14" comment
    blocks at L1298-1303 + L1357-1359 and the matching note in
    `simulation_detection_probability.py:619-623`. Added
    `TestPDetGridMassCoordinateFrame` property tests.
- **Reference:**
  - Mandel, Farr & Gair (2019) arXiv:1809.02063 §2 — selection function
    in population inference is evaluated at hypothesis (candidate)
    parameters, not at the measured data.
  - Loredo (2004) arXiv:astro-ph/0409387 — proper Bayesian treatment
    of selection effects.
  - Maggiore (2008) Vol 1 §4.1.4 — observer-frame redshifted mass
    `M_z = M_source · (1+z)` enters waveform amplitude.
  - Babak et al. (2017) arXiv:1703.09722 §III — EMRI waveform
    parametrization in observer frame.
- **Result (job 4252817 cpu_il, 2026-05-06; reusing Tier-3+bridge prepared CRB at h=0.73):**

  | Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z |
  |---|---|---|---|---|---|
  | 1D | 0.7300 | **0.7309** | 0.0047 | +0.0009 | **+0.18σ PASS ✅** |
  | 2D | 0.7300 | **0.7307** | 0.0037 | +0.0007 | **+0.20σ PASS ✅** |

  - 2D bias dropped from +0.0141 to +0.0007 — **20× reduction**.
  - 1D channel unchanged (already clean post-bridge); the H3 fix is
    a no-op in the 1D path because the 1D grid has no M axis.
  - **2D bias (+0.0007) now ≤ 1D bias (+0.0009)** — info monotonicity
    restored; the BH-mass channel correctly tightens the posterior.
  - 2D σ_boot (0.0037) tighter than 1D σ_boot (0.0047), as expected
    when the 2D channel adds genuine information.
- **Status:** code change committed (`f01595c`); R1 cluster validation
  PASS at h=0.73 phase46-merged 1473 events. R2 (Phase 45 412 events)
  shows 1D-2D asymmetry resolved (1D z=+2.40σ, 2D z=+3.20σ;
  Δ=0.0007); residual +0.0125 bias on Phase 45 attributable to
  combined bridge+H3 effects on the smaller dataset (within
  seed-dependent MAP scale of 0.02 per `finding_seed_dependent_map`).
  Multi-truth panel re-run on phase46-merged remains as a follow-up
  phase per the original plan.
- **Note on Phase 14 supersession:** the "p_det uses ML mass estimate
  (detection.M) rather than M_gal*(1+z)... known approximation, not a
  bug" comment from Phase 14 is now formally superseded by the
  hypothesis-evaluation principle. The approximation was justified
  while σ_boot was wide enough (>~0.005) to mask the residual; it
  failed under tighter post-bridge σ_boot=0.0039 on 1473 events.
- **Detail →** `.planning/2D-CHANNEL-AUDIT-20260505.md` Step 4 (post-fix)
  and `~/.claude/plans/please-look-at-the-velvety-quail.md` for the
  full plan-to-validation narrative.

### 3.16 M1-prior emcee under-mixing → seed-dependent (M, z) marginal (2026-05-16)

- **Symptom:** the production CRB at
  `simulations/cluster_run_production_h0p73_20260506/simulations/prepared_cramer_rao_bounds.csv`
  (1549 events, h=0.73) showed a sharp **row-424 boundary** between two
  mass libraries: pop A (rows 0–423) dominated by M ≈ 4.6e5 (median
  M = 3.7e5, median d_L = 0.32 Gpc, median SNR = 24), pop B (rows
  424–1549) dominated by M ≈ 2.2e5 (median M = 2.2e5, median d_L =
  0.07 Gpc, median SNR = 31). The bi-modality drove an elbow at N≈400
  in the M_z-improvement HDI68-vs-N convergence curve
  (`figures/paper_m_z_improvement.pdf`) that masqueraded as "data
  became more informative" when it was actually concatenation order.
- **Mechanism (cluster forensics, 2026-05-16):**
  1. **Production CRB is a concatenation of two simulation campaigns.**
     `run_20260401_seed200/simulations/cramer_rao_bounds.csv` (SNR_THR=15,
     4497 raw events; SNR≥20 subset = 424 events) followed by
     `run_20260504_seed300_extension/simulations/cramer_rao_bounds_simulation_*.csv`
     (SNR_THR=20, 1050 events). The M=463534 count in production rows
     0–423 matches seed200's SNR≥20 subset to the unit (176 events
     each); top-5 M-libraries match in both directions. ~75 events
     (rows 1475–1549) remain unaccounted, likely a third small
     extension or rerun, not yet pinned down.
  2. **`cosmological_model.py` sampler code is unchanged across the
     campaign window** (Apr 7 → May 4). `git log` on
     `cosmological_model.py`, `M1_model_extracted_data/`, and
     sampler constants (nwalkers=20, burn_in=1000) shows only refactor
     commits. PE-01 (commit `55a6d99`) threads `h_inj` into
     `set_host_galaxy_parameters` but is a no-op at h_inj=H=0.73 by
     construction. Same code, same prior, different seeds.
  3. **`Model1CrossCheck.setup_emri_events_sampler` was under-mixed.**
     Measured integrated autocorrelation time
     τ_ACT ≈ 33 steps for both log10(M) and z. Prior burn_in_steps=1000
     sat at 30·τ_ACT, below the 50·τ_ACT safety margin in Foreman-Mackey
     et al. (2013) §4. Per-task chains converged to seed-dependent
     equilibria within the M1 prior (R_emri(M) × dN/dz(M, z) has a
     sharp R_emri peak at M ≈ 2.5e5 plus long tails to [3e4, 1e6]).
  4. **Selection effect amplification.** Per-task SLURM jobs consume
     only the first ~25 events of the 4000-sample batch returned by
     `sample_emri_events(200)` before exiting, so the burn-in
     equilibrium dominates the per-task (M, z) marginal. The downstream
     SNR≥20 cut amplifies whatever sampling bias exists: a mild ~10%
     cross-seed median-M drift becomes a ~2× apparent split downstream
     because heavy MBHs at d_L ≈ 0.3 Gpc survive the SNR cut just barely
     (SNR ≈ 22) while light MBHs need d_L ≲ 0.1 Gpc (SNR ≈ 30+) — the
     selection function reshapes seed200's high-M chain into the heavy
     tail and seed300's lower-M chain into the close/light region.
- **Diagnostic (inline, no scripts):**
  ```python
  # Cluster, via awk:
  awk -F, 'NR>1 && $122>=20{printf "%.0f\n", $1}' \
    /pfs/work9/.../run_20260401_seed200/simulations/cramer_rao_bounds.csv \
    | sort | uniq -c | sort -rn | head
  # → 176×463534, 52×318738, 43×294877, ... (matches production rows 0–423)

  # Local, autocorrelation measurement:
  sampler.run_mcmc(p0, 5000); sampler.get_autocorr_time(quiet=True)
  # → τ ≈ 33 steps for log10(M), τ ≈ 34 for z
  ```
  Cross-seed median-M ratio (per-task seeds 200–219 vs 300–319,
  single-batch reproduction): OLD config 0.91, NEW config 1.02.
- **Fix (commit `991333a`, this file change):**
  - `master_thesis_code/cosmological_model.py:setup_emri_events_sampler`:
    `nwalkers = 20 → 50`, `burn_in_steps = 1000 → 10000`. New values
    give 10000 ≈ 300·τ_ACT burn-in plus 2.5× walker coverage.
  - Cost: integration test fixture burn-in goes ~3 s → ~30 s (slow tests
    only). Smoke run confirms cross-seed log10(M) std ≈ 0.35 (broad
    coverage), within-campaign std halved (1.74e4 → 1.17e4).
- **Impact on H0 — none.** d_L–z relation per event and event-by-event
  likelihoods are unaffected by M-marginal heterogeneity. The bootstrap
  σ_boot remains a correct variance estimator. The convergence-curve
  elbow at N≈420 in `paper_m_z_improvement.pdf` is purely a
  concatenation-order artifact. The user's original hypothesis (M vs
  M_z simulation bug) was quantitatively disproven before this
  diagnosis: dividing pop A's M by (1 + z_est) leaves a 51% residual
  gap from pop B's M, ruling out a missing (1+z) Jacobian. The actual
  cause turned out to be sampler under-mixing, not a physics-formula
  bug.
- **Reference:**
  - Foreman-Mackey, Hogg, Lang & Goodman (2013) "emcee: The MCMC
    Hammer" arXiv:1202.3665 §4 (burn-in ≥ 50·τ_ACT, nwalkers ≥ 2·ndim
    with margin).
  - Goodman & Weare (2010) "Ensemble samplers with affine invariance"
    — original affine-invariant ensemble sampler.
- **Paper policy:** keep the merged 1549-event production CRB as the
  paper-grade dataset. Disclose in methods/appendix that the CRB is
  `seed200(SNR≥20) ⊕ seed300_extension(SNR≥20)` concatenated in that
  order; document the per-task emcee under-mixing observation;
  demonstrate H0-MAP robustness across (merged, seed200-only,
  seed300-only) subsets — predicted invariance within σ_boot since the
  d_L–z inference channel is unaffected. The sampler fix applies to
  any future `python -m master_thesis_code <dir> --simulation_steps N`
  campaign run.
- **Status:** sampler fix committed (`991333a`); production CRB
  preserved; no re-run scheduled. Pinning down the residual ~75
  events between row 1474 and 1549 remains open as a low-priority
  pre-submission task.
- **Detail →** `.planning/HANDOFF-PLOTTING-OVERHAUL-20260516.md` §A
  (full forensics + paper policy) and
  `memory/project_crb_two_population.md` (concise memory entry).

### 3.12 Galactic confusion noise in PSD (Phase 9)

- **Symptom:** Pre-fix LISA PSD lacked the galactic foreground component;
  SNR estimates unrealistically high; Fisher-matrix uncertainties biased low.
- **Mechanism:** PSD missing galactic-confusion term per Babak et al. (2023)
  Eq. 17, with observation-time-dependent knee frequency.
- **Diagnostic:** PSD comparison against Babak et al. reference.
- **Fix:** Commit `3bed9fc` — `_confusion_noise()` added to `LisaTdiConfiguration`.
- **Evidence:** SNR distribution shifted to realistic regime; Fisher
  uncertainties broadened to physical values.
- **Reference:** Babak et al. (2023) arXiv:2303.15929 Eq. 17.
- **Detail →** [Appendix A · Phase 9](#phase-9--galactic-confusion-noise-2026-03-29).

---

## 4. Open Issues & Active Work

### 4.0 Continuation guide — pick this up cold in another session

**If you're starting fresh, read these in order (post-H3-fix state, 2026-05-06):**
1. §1 Executive Summary, §2 Current Cluster Numbers — current state in 2 minutes.
2. **§3.15 H3 fix** (LANDED 2026-05-06 commit `f01595c`) — numerator
   p_det query: observation→hypothesis + 2D grid M_z axis. **Fully
   closed 2D channel** at h=0.73 phase46-merged 1473 events (1D z=+0.18,
   2D z=+0.20, info monotonicity restored).
3. **§3.14 bridge fix** (LANDED 2026-05-05 commit `2b33cad`) — replaces
   the Phase 45 anchor scheme with a principled monotonic-asymptotic
   bridge; closed 1D channel and reduced 2D z from +37 to +3.60. The
   §3.14 bridge handles the d_L axis; §3.15 H3 handles the M axis +
   numerator hypothesis convention. Together they form a triple with
   §3.13 (Tier 3 D(h) double-counting) for full 2D-channel correctness.
4. §3.13 (Tier 3 D(h) fix, LANDED 2026-05-04) and §3.5 (Phase 45 anchor,
   superseded by §3.14) for prior context.
5. `~/.claude/plans/please-look-at-the-velvety-quail.md` — Phase 47 H3
   plan (now landed); kept for the plan-to-validation narrative.

**Done through 2026-05-06:**
A0 G0a CLEAN · A1 G1b real shift · A2 G2b 1D over-anchors · A4 G4b D(h)
DOMINATES bias · A5 G5a-PARTIAL · A8 G8a both verdicts hold ·
A7-redux FAIL (fine grid: 1D z=+5.62σ, 2D z=+3.36σ; withdrew G7a UNBIASED) ·
Tier 2 bootstrap-subsample at h=0.73 confirmed bias is structural ·
Tier 3 audit identified D(h) double-application via test_21+test_22 ·
**Tier 3 FIX LANDED**: closure h=0.65 1D z=+1.67σ PASS, 2D z=+1.68σ PASS;
production h=0.73 1D z=+1.4σ PASS, 2D z=+1.97σ PASS.
**Bridge fix LANDED (2026-05-05, commit `2b33cad`)**: principled
monotonic-asymptotic out-of-grid extrapolation for both 1D and 2D p_det
channels (uniform scheme replacing the Phase 45 anchor + raw scipy
extrapolation); 1D channel closed at h=0.73 phase46-merged 1473 events
(z=+0.19); 2D residual dropped 10× (z=+37 → +3.60) but did not fully
close. **H3 fix LANDED (2026-05-06, commit `f01595c`)**: numerator
p_det query observation→hypothesis + 2D grid M_z axis, see §3.15.
2D channel closed at h=0.73 phase46-merged 1473 events (z=+0.20σ,
bias=+0.0007 ≤ 1D bias=+0.0009, info monotonicity restored).

Diagnostic scripts at `scripts/bias_investigation/test_{13,14,16,19,20,21,22}_*.py`.
Cluster CSVs at `simulations/cluster_run_phase45_20260501/` and
`simulations/cluster_run_closure_h065_20260503_finegrid/posteriors{,_with_bh_mass}/`.

**Tier 3 finding (now resolved): D(h) was applied twice.**

Phase 32 placed `/D` inside L_comp = num/D (correct: prior normalization for
`p_galaxy ∝ p_det · dV_c` per Gray Eq. 31). Phase 43-H1 added an outer
−N · log D in `combine_log_space` (incorrect: re-applying selection
correction). Tier 3 fix removes the outer correction. See §3.13 catalog entry.

**Next actions (post-H3-fix, 2026-05-06):**

| # | Action | Cost | Blocks |
|---|---|---|---|
| 1 | Multi-truth panel re-run at h ∈ {0.60, 0.65, 0.70, 0.73} on phase46-merged under combined Tier-3+bridge+H3 fixes. | ~1 hr cluster | Paper-grade multi-truth verification across the panel. |
| 2 | Update paper with post-H3-fix MAP values and σ_boot (1D MAP=0.7309, 2D MAP=0.7307; both within 2σ of truth). | ~few hr | Submission. |
| 3 | (Lower priority) DELETE legacy Plan 45-06 / 45-07 side branches; both superseded by §3.14 + §3.15. | n/a | Tree cleanup. |
| 4 | (Optional) `extract_baseline()` / `TRUE_HUBBLE_CONSTANT` cleanup per §4.3, §4.4. | ~30 min | Footgun cleanup, not paper-blocking. |

**Context for cold pickup:**
- Project root: `/home/jasper/Repositories/MasterThesisCode`
- Cluster: bwUniCluster 3.0 (KIT), GPU partition. SSH access available
  via `ssh bwunicluster`. SSH agent set up for git push.
- Current working tree state: branch `main`, two staged side branches
  (`phase-45-option-A`, `phase-45-option-D`) — **DO NOT MERGE** per A4
  finding.
- A parallel session began running 45-06 and 45-07 on the cluster (per
  user 2026-05-01); the cluster results from those runs will inform but
  not deploy — they're empirical data points, not validations.
- Diagnostic outputs: `scripts/bias_investigation/outputs/phase45/*.json`.
- Cluster CSVs rsync'd to: `simulations/cluster_run_phase45_20260501/`
  (cramer_rao_bounds.csv, prepared_cramer_rao_bounds.csv,
  diagnostics/event_likelihoods.csv, fisher_quality.csv).

### 4.1 Cluster MAP residual +0.025 (Phase 45 ESCALATING)

Phase 45 has shipped two anchor escalations (Plans 45-02 single anchor,
45-04 hybrid). Cluster MAP moved from 0.7650 to 0.7550 — one discrete grid
step toward truth, residual still +0.025 (3 σ_boot).

**Two further escalations are STAGED but DEPLOYMENT-BLOCKED:**

| Plan | Branch | Commit | Description | Status |
|---|---|---|---|---|
| 45-06 (Option A) | `phase-45-option-A` | `ca64ffb` | Raise d_L=0 anchor 0.7931 → 0.8873 (point estimate) | **BLOCKED on A4 + A7** (A0–A2, A5, A8 done; A2 G2b suggests direction may be wrong) |
| 45-07 (Option D) | `phase-45-option-D` | `12d3ebc` | Extend hybrid anchor to `_build_grid_2d` (with_bh_mass channel) | **BLOCKED on A4 + A7** (A2 cluster-projection shows 1D over-anchors by Δ=−0.27; extending to 2D risks over-correction) |

**Reason for block:** ten concerns surfaced in critical review; four most
serious are:
1. No closure test on a non-0.73 truth — calibration could be tuned.
2. Phase 45 anchor is being empirically tuned, not principled.
3. Discrete Δh=0.005 + σ_boot ≈ 0.01 makes shifts uninterpretable.
4. with_bh_mass MAP closer to truth (0.7450) than the patched 1D channel
   (0.7550) — suggests 1D anchor may be overshooting.

### 4.2 v2.3 Audit Programme (active)

Decision-tree of eight prioritized audits (full plan:
`/home/jasper/.claude/plans/can-you-critically-think-calm-cocke.md`):

| ID | Audit | Cost | Gate | Status |
|---|---|---|---|---|
| **A0** | Hubble-constant consistency (`H` vs `TRUE_HUBBLE_CONSTANT`) | read-only 15 min | G0a clean / G0b bug | **DONE — G0a CLEAN** (2026-05-01) |
| **A1** | Δh=0.001 reinterpretation of existing posteriors | no cluster, 1 hr | G1a/b/c per continuous MAP | **DONE — G1b** (2026-05-01): continuous 1D MAP=0.7550, σ_boot=0.0109, real shift |
| **A2** | 1D vs 2D channel structural audit (`p̂(c_0)` per channel, lift comparison) | no cluster, 30 min | G2a/b/c per anchor signedness | **DONE — G2b cluster-projection** (2026-05-01): 1D anchored mean lift on [0, c_0_cluster=0.10] ≈ 0.834 vs 2D unanchored ≈ 0.568 (Δ=−0.27, sensitivity-robust). Local data limited by geometry mismatch (local dl_max=2.77 Gpc vs cluster 12 Gpc). |
| **A3** | Sub-binning quantitative pre-prediction (RESEARCH §4b) | no cluster, 30 min | G3a/b/c/d per predicted MAP shift | DEFERRED (needs cluster-scale data; local geometry mismatch ↔ A2) |
| **A4** | Per-event L_cat / L_comp / f_i CSV analysis at 412 events | 0–15 min cluster | G4a/b/c per bias concentration | **DONE — G4b + decomposition** (2026-05-02): rsync'd cluster CSVs to `simulations/cluster_run_phase45_20260501/`. Reconstructed cluster MAP=0.7550 exactly. **Critical finding: Σ log L_i alone peaks at h=0.7400 (within σ_boot of truth); D(h) selection-function correction shifts MAP +0.015 → 0.7550.** D(h) effect is 2.7× per-event L pull, opposite sign. Anchor escalation aimed at wrong layer. |
| **A5** | Phase 43 H1/H2 ablation evidence audit | read-only 30 min | G5a/b per ablation evidence presence | **DONE — G5a-PARTIAL** (2026-05-01): H2 alone moved MAP 0.860 → 0.730 ("raw Σ log L_i already peaks at 0.730 without needing D(h) correction"); H1 only "sharpens the peak". H1 standalone effect not tested but inferred small. |
| **A6** | Native Δh=0.001 cluster sweep | 30 min cluster | G6a/b vs A1 interp agreement | RESERVED (only if A1 contested; A1 cubic cross-check passed) |
| **A7** | Closure test at h_true=0.65, 4-h-grid (initial) | ~12 min cluster (lean rescaling, 4-h-grid on dev_cpu_il, job 4198463) | G7a/b/c/d per recovered MAP | **WITHDRAWN — G7a was a smoothing artifact** (2026-05-03 → 2026-05-04). Cubic spline through 4 widely-spaced points (Δh=0.05) with truth on grid centre cannot resolve a sub-grid bias of ≲0.025. Re-evaluated by A7-redux. |
| **A7-redux** | Closure test fine grid (11 h-values, Δh=0.005, bootstrap σ_boot) | 30 min cluster (job 4200482, dev_cpu_il, 4-task chained array) | PASS \|MAP-0.65\|≤3·σ_boot / FAIL otherwise | **DONE — FAIL** (2026-05-04): 1D MAP=0.6708 bias=+0.0208 σ_boot=0.0037 z=+5.62σ; 2D MAP=0.6697 bias=+0.0197 σ_boot=0.0059 z=+3.36σ. Δ(−N log D)=+7.6 dominates ΔΣ log L=−3.6. **D(h) structural bias confirmed empirically across two truths.** |
| **Tier 2** | Bootstrap-subsample at h=0.73 | local, no cluster | sample-fluctuation vs systematic | **DONE — SYSTEMATIC** (2026-05-04, `test_19_bootstrap_subsample.py`): σ_boot=0.0102 stable across N ∈ {200, 300, 412}, no convergence toward truth as N grows; full-sample MAP at median of bootstrap distribution. +0.025 is NOT sample fluctuation. |
| **A8** | Phase 33/34 verdicts revisit against current baseline | read-only 45 min | G8a/b per test type | **DONE — G8a both** (2026-05-01): Phase 33 "zero delta vs 30-bin baseline, log-posteriors identical to 4 decimal places" — baseline-invariant test; Phase 34 0 events excluded by condition filter — `allow_singular` removal a no-op. Both verdicts hold. |

**Stopping condition for paper (revised 2026-05-04):** Tier 3 D(h) audit
must identify the systematic in `precompute_completion_denominator()`,
either (a) fix it and re-run cluster eval at h=0.73 + closure at h=0.65 to
confirm both pass, OR (b) document the systematic with quantitative bound
σ_sys ≥ 0.025 and quote the result as `h = h_MAP ± σ_stat ± σ_sys`. The
original "A7 PASS PLUS A1 continuous MAP" gate is no longer applicable
since A7-redux fails.

### 4.3 A0 result (2026-05-01) — `TRUE_HUBBLE_CONSTANT=0.7` is dead code

Audit A0 verdict **G0a CLEAN**: production paths use `GalaxyCatalogueHandler`
(in `master_thesis_code/galaxy_catalogue/handler.py`) and pass the active
`h_value` from `--h_value` CLI flag end-to-end. The stale constant
`TRUE_HUBBLE_CONSTANT=0.7` (`master_thesis_code/constants.py:26`) only flows
into:
- `GalaxyCatalog.__init__` default (`datamodels/galaxy.py:104`) — synthetic
  catalog only; Pipeline A was deleted in commit `c1571a2`.
- `GalaxyCatalog._build_comoving_volume_element_spline` (`galaxy.py:115`).
- `GalaxyCatalog.get_possible_host_galaxies` (`galaxy.py:366`).
- `master_thesis_code_test/test_benchmarks.py:11,16,17` — a benchmark test only.

**Latent footgun (low-priority cleanup):** `TRUE_HUBBLE_CONSTANT=0.7` differs
from production injection truth `H=0.73`. Any future code that instantiates
`GalaxyCatalog` or invokes the static spline method without passing `h0=`
will silently get the wrong default. Recommended action: delete
`TRUE_HUBBLE_CONSTANT` and the residual `GalaxyCatalog` synthetic class if
Pipeline A is truly gone, or align both to 0.73. Tracked but not blocking.

### 4.4 `extract_baseline()` deprecated by warning, not deletion

Phase 43 deprecated `extract_baseline()` with a logger.warning citing the
missing `−N log D(h)` term. The function is **still callable**. Future code
that calls it will get a biased MAP with only a log warning. Recommended:
either delete or make it raise. Footgun. Tracked, not blocking.

### 4.5 Phase 43 H1/H2 ablation evidence (Audit A5 — DONE 2026-05-01)

**Verdict: G5a-PARTIAL — concern reasonably addressed.**

Phase 43 SUMMARY (43-03-SUMMARY.md:49) explicitly documents H2's standalone
effect:

> "The H2 CRB fix (equatorial→ecliptic frame migration) was the primary
> driver. With correct sky angles, BallTree now matches EMRI events to
> their true host galaxies. **The host likelihoods L_i(h) now peak sharply
> at h_true=0.73, and the raw Σ log L_i already peaks at 0.730 without
> needing D(h) correction.**"

So H2 alone moved MAP from 0.860 to 0.730 (the full shift). H1 only
"sharpens the peak" via the D(h) selection function — secondary effect.
H1's standalone effect (without H2) wasn't directly tested but is inferred
to be small from this evidence.

Host recovery improvement (31→38/60) and "no possible hosts" (10→1) provide
independent corroboration of H2's primary role.

### 4.6 Phase 33/34 verdicts vs current baseline (Audit A8 — DONE 2026-05-01)

**Verdict: G8a — both verdicts hold.**

- **Phase 33 (P_det grid resolution):** 33-02-SUMMARY.md:28 reports "Zero
  delta vs 30-bin baseline — log-posteriors identical to 4 decimal places
  across all 38 h-values, both variants". This is a "MAP delta from baseline"
  test, baseline-invariant. The verdict "P_det grid resolution is not a
  source of bias" remains valid against any baseline.
- **Phase 34 (Fisher quality):** 34-02-SUMMARY.md confirms 0 events were
  excluded by the Fisher condition-number filter at threshold 1e10 (empirical
  cond_4d range 2.5e8–5.2e14 ⇒ all events retained); the `allow_singular=True`
  removal had no observable MAP impact. The infrastructure (fisher_quality.csv,
  comparison report section) is wired for future use but the current dataset
  doesn't trigger any exclusion. Verdict holds.

No re-runs required. Both phases dropped from "concern #6" follow-up.

### 4.7 H3 — 2D-channel residual fix (LANDED 2026-05-06, see §3.15)

**Status:** Code change committed (`f01595c`); R1 cluster validation
PASS. Promoted to catalog entry §3.15. The remainder of this section
is preserved as the planning-state record for cold-pickup readers
who want to see the diagnosis-before-fix narrative.

**Symptom (post-bridge fix `2b33cad`, 2026-05-05):** the 1D channel at
h=0.73 phase46-merged 1473 events PASSES at z=+0.19 (MAP=0.7309 within
0.001 of truth — cleanest 1D dark-siren H₀ recovery to date). The 2D
channel residual is z=+3.60 (bias=+0.0141, **16× larger than 1D bias**).
Adding the BH-mass likelihood (channel B) is *adding information*
relative to position-only (channel A); per information monotonicity
that should *tighten* the posterior, not move it further from truth.
2D bias > 1D bias is therefore a structural 2D-specific bug, not a
statistical residual.

**Bug identification (during 2026-05-05 plan-mode review with the user):**
the 2D channel has **two coupled issues**, not one. The original
handoff `.planning/HANDOFF-2D-RESIDUAL-H3-INVESTIGATION-20260505.md`
framed H3 only as a "grid axis vs query coordinate mismatch
(M_source vs M_z)". The user pushed back on the proposed grid-rebuild
fix and surfaced the deeper layer:

1. **Numerator queries the observation, not the hypothesis.** At
   `bayesian_statistics.py:1304-1306` the integrand passes
   `np.full_like(z, _det_M)` — the detection's measured ML
   observer-frame mass, **constant across integration over candidate
   redshift z**. But the integrand's *hypothesis* at each integration
   z is "the source is the host candidate at z with source-frame mass
   `host_M`", which implies observer-frame `M_z = host_M · (1 + z)`,
   **varying with integration z**. The rest of the integrand already
   uses the hypothesis (lines 1322-1325 compute
   `mu_gal_frac = host_M·(1+z) / _det_M` and feed the Gaussian
   product at the observed `_det_M` via the Fisher covariance).
   Only the p_det query is using the observation. Phase 14's
   "approximation, not a bug" comment was justifying this exact
   mismatch; under tighter post-bridge σ_boot it is no longer harmless.
2. **Grid axis vs query coordinate mismatch.** The grid is built in
   source-frame `M` (`_M_arr` from injection CSV) but queries pass
   observer-frame `M_z` (denominator: `M·(1+z)`; numerator: `_det_M`
   which is M_z). At z≈0.5 the queries land ~50% higher than the
   grid bin labels.

These two issues are **coupled** — really one bug expressed across
the choice of mass coordinate. Fixing (1) without (2) leaves queries
inconsistent with the grid axis; fixing (2) without (1) leaves the
numerator using the wrong physical quantity.

**Planned fix (Option A — observer-frame M_z everywhere):**

| Where | Currently | Planned |
|---|---|---|
| 2D grid axis | source-frame `M` (`_M_arr`) | observer-frame `M_z = _M_arr · (1 + _z_arr)` |
| Numerator query (L1304) | `np.full_like(z, _det_M)` (observation, z-constant) | `host_M · (1 + z)` (hypothesis, varies with integration z) |
| Denominator query (L1361) | `M · (1 + z)` (already correct) | unchanged |

Option A is preferred over the equivalent Option B (source-frame `M`
everywhere; numerator passes `host_M`, denominator passes `M`)
because (i) `M_z` is the natural physical SNR-determining coordinate;
(ii) the hypothesis-redshift relationship is at the call site,
matching the GW likelihood's mu_gal_frac convention; (iii) it removes
the implicit injection-z encoding in the grid axis that has been a
Phase-14 stumbling block.

**Implementation steps (per the plan):**

1. **Diagnostic probe** — `scripts/bias_investigation/test_27_m_coordinate_mismatch.py`
   reconstructs the numerator integrand offline at sample integration z
   for ~1500 events at h=0.73; reports per-event Δlog L_i and predicted
   joint MAP shift Δh. Decision gate G_H3a: PASS if Δh has correct
   sign and magnitude ≈ −0.013 (closing the +0.0141 residual).
2. **Production change** under `/physics-change` protocol — modify
   `_get_or_build_grid` (one line) and the numerator at
   `bayesian_statistics.py:1304-1306`. Add three new property /
   regression tests. Remove the "approximation, not a bug" comment
   blocks. `[PHYSICS]` commit.
3. **Cluster validation** (narrow scope per user choice 2026-05-05):
   - **R1** primary gate: h=0.73 phase46-merged 1473 events on cpu_il
     (~15 min). PASS iff 2D z ≤ 2σ AND 2D bias ≤ 1D bias.
   - **R2** Phase 45 412-event regression check (local ~5 min); also
     clears the bridge fix's pending Phase 45 re-validation.
   - Multi-truth panel (h=0.60/0.65/0.70) **deferred to a follow-up
     phase** contingent on R1 passing.
4. **Documentation** — promote this §4.7 entry into a §3.15 catalog
   entry once the fix lands and post-fix numbers are available.

**References for the post-fix derivation:**
Mandel, Farr & Gair (2019) arXiv:1809.02063 §2 (selection function
evaluated at hypothesis); Loredo (2004) arXiv:astro-ph/0409387
(proper Bayesian treatment of selection effects); Maggiore (2008)
Vol 1 §4.1.4 (observer-frame redshifted mass `M_z = M · (1+z)`);
Babak et al. (2017) arXiv:1703.09722 §III (EMRI waveform
parametrization in observer-frame).

**Why not part of §3.14 (the bridge fix):** §3.14 fixed the
**d_L axis** extrapolation (out-of-grid policy at d_L<dl_min and
d_L>dl_max). §3.15 fixes the **M axis** coordinate convention plus
the numerator's observation-vs-hypothesis convention. They are
independent layers of the 2D channel; both are needed to close the
residual. Together with §3.13 (Tier 3 D(h) double-counting) they
form the three legs of the 2D-channel correctness triple.

**Foot-gun risk if implemented incorrectly:** a subtle failure mode
is rebuilding the grid in M_z but NOT changing the numerator query —
this fixes (2) but not (1) and would leave a residual smaller than
+0.0141 but still nonzero. The plan's diagnostic gate G_H3a guards
against this by predicting Δh under the *full* fix (both layers);
if the real post-fix MAP doesn't match the prediction, that's the
signal that one layer was missed.

**Plan artifact:** `~/.claude/plans/please-look-at-the-velvety-quail.md`.
**Memory:** `project_pdet_hypothesis_convention.md` (saved 2026-05-05).

---

## 5. Eliminated Hypotheses

| Hypothesis | Phase | Verdict | Evidence |
|---|---|---|---|
| Fisher matrix accuracy | Phase 10 | Fixed (5-point stencil), insufficient alone | [§3.11](#311-five-point-fisher-derivatives-phase-10) |
| PSD missing confusion noise | Phase 9 | Fixed, insufficient alone | [§3.12](#312-galactic-confusion-noise-in-psd-phase-9) |
| KDE P_det artifacts | Phase 11.1 | Fixed (KDE → IS), insufficient alone | [§3.10](#310-kde--simulation-based-is-p_det-estimator-phase-111) |
| Gaussian evaluation index bug | Investigation 2026-03-31 | NOT A FACTOR (delta-function approx makes 3D vs 4D negligible) | [Appendix A](#gaussian-index-bug-investigation-2026-03-31) |
| Spurious /(1+z) Jacobian | Phase 15 | Fixed, insufficient alone | [§3.7](#37-spurious-1z-jacobian-phase-15--minor-insufficient-alone) |
| P_det grid boundary/construction (KDE-era) | Phases 17–20 | VALIDATED CORRECT (VALD-01/02 PASS) | [Appendix A](#phases-1720--injection-campaign-validation-2026-03-31-to-2026-04-01) |
| Numerical posterior underflow | Phases 21–23 | Fixed (log-space + 4 strategies), not the cause | [Appendix A](#phases-2123--posterior-numerical-stability-2026-04-02) |
| P_det extrapolation zeros (fill_value=0.0) | v1.4 | Fixed, partial improvement (-9.2% → -6.9%) | [§3.9](#39-p_det-extrapolation-fill_value00-v14--commit-44d5358) |
| L_comp local-window normalization | Phase 32 | Fixed, MAP 0.60 → 0.73 local | [§3.1](#31-local-window-l_comp-denominator-phase-32--map-013-local) |
| P_det grid resolution (30 vs 60 bins) | Phase 33 | Validated — not a bias source (Audit A8 G8a 2026-05-01: zero MAP delta is baseline-invariant; verdict holds) | [Appendix A · v2.1](#v21-h0-bias-resolution-milestone-shipped-2026-04-09) |
| Fisher matrix `allow_singular=True` | Phase 34 | Fixed, condition gate added | [Appendix A · v2.1](#v21-h0-bias-resolution-milestone-shipped-2026-04-09) |
| Equatorial GLADE / EMRI ecliptic frame mismatch | Phase 36 + 43-H2 | Fixed, host recovery 31→38/60 | [§3.2](#32-equatorial-crbs-vs-ecliptic-glade-phase-36--phase-43-h2--host-recovery-313860) |
| Per-parameter Fisher epsilon | Phase 37 PE-02 | Fixed | [§3.8](#38-per-parameter-fisher-epsilon-phase-37-pe-02) |
| L_cat formula (Gray Eqs. 24–25) | Phase 38 | Fixed | [§3.6](#36-l_cat-formula-phase-38--gray-eqs-2425-normalization) |
| `extract_baseline` missing -N log D(h) | Phase 43-H1 | Fixed | [§3.3](#33-extract_baseline--combine-missing--n-log-dh-phase-43-h1--removes-monotone-h-pull) |
| Cached equatorial CRBs on disk | Phase 43-H2 | Fixed | [§3.2](#32-equatorial-crbs-vs-ecliptic-glade-phase-36--phase-43-h2--host-recovery-313860) |
| h-dependent P_det zero-fill at c_0 ∝ 1/h | Phase 44 | Fixed, cluster 0.860 → 0.7650 | [§3.4](#34-h-dependent-p_det-zero-fill-cutoff-at-c_0--1h-phase-44--cluster-0860--07650) |
| First-bin asymptote underestimate (single-anchor) | Plans 45-02/03 | Fix shipped, sub-grid-step on production | [§3.5](#35-p_det-first-bin-asymptote-underestimate-at-d_l--c_0-phase-45-active) |
| First-bin asymptote underestimate (hybrid anchor) | Plans 45-04/05 | Fix shipped, 1 grid step toward truth | [§3.5](#35-p_det-first-bin-asymptote-underestimate-at-d_l--c_0-phase-45-active) |
| H=0.73 vs TRUE_HUBBLE_CONSTANT=0.7 inconsistency | Audit A0 | NOT A PRODUCTION PATH (G0a clean) | [§4.3](#43-a0-result-2026-05-01--true_hubble_constant07-is-dead-code) |

---

## 6. Reproducibility Recipes

### 6.1 Reproduce current cluster MAP (--evaluate path, 60 events local)

```bash
cd /home/jasper/Repositories/MasterThesisCode
uv run python -m master_thesis_code simulations/ \
    --evaluate --h_value 0.73 --log_level INFO
```

Expected: MAP=0.730 (Phase 43 VERIFY-03 SC-3 PASS). Reads
`simulations/prepared_cramer_rao_bounds.csv` (60-event SNR≥20 subset).

### 6.2 Reproduce cluster injection campaign + evaluation (full pipeline)

```bash
# Injection campaign (1 task example; production uses array)
uv run python -m master_thesis_code <workspace> \
    --injection_campaign --simulation_steps 412 \
    --h_value 0.73 --seed 200 --simulation_index 0

# Cluster pipeline:
#   sbatch cluster/submit_injection.sh      # full injection grid
#   sbatch cluster/merge.sbatch             # emri-merge + emri-prepare
#   sbatch --array=0-37 cluster/evaluate.sbatch  # 38 H_VALUES, Δh=0.005 peak
#   sbatch cluster/combine.sbatch           # combine per-h posteriors
```

Phase 45 cluster re-eval template: `cluster/submit_phase45_eval.sh`
(reuses existing CRBs, reruns evaluate + combine).

### 6.3 Run the diagnostic battery (T1–T12)

```bash
cd /home/jasper/Repositories/MasterThesisCode
for n in 01 02 03 04 05 05b 07 08 09 10 11 12; do
    uv run python "scripts/bias_investigation/test_${n}_*.py"
done
```

Outputs land at `scripts/bias_investigation/outputs/phase45/` as JSONs.
Per-test scope: see [§3.5](#35-p_det-first-bin-asymptote-underestimate-at-d_l--c_0-phase-45-active).

### 6.4 Audit A1 — Δh=0.001 reinterpretation

```bash
uv run python scripts/bias_investigation/test_13_fine_grid_map.py
```

Reads cached per-event likelihoods from `results/phase45_v2_posteriors/` (1D)
and joint posterior from `results/phase45_v2_posteriors_with_bh_mass/` (2D);
linearly + cubic-spline interpolates `log L_i(h)` and `log D(h)` to a
Δh=0.001 grid in [0.60, 0.86]; recomputes `joint log p(h) = Σ log L_i − N log D`;
finds continuous MAP, posterior mean, 68%/95% HPD; bootstraps B=1000 (RNG
seed 20260501) on the fine grid for 1D.

Outputs:
- `scripts/bias_investigation/outputs/phase45/fine_grid_map.json`
- `scripts/bias_investigation/outputs/phase45/fine_grid_map_1d.png`

Pre-registered gates G1a (truth recovered) / G1b (real shift) / G1c (robust
bias > 3σ_boot) defined in script docstring.

**Result (2026-05-01):** G1b — continuous MAP=0.7550 (lin) / 0.7540 (cub),
σ_boot=0.0109, |cubic-linear|=0.0010 PASS. Bias is genuine.

### 6.5 Audit A2 — 1D vs 2D channel structural audit

```bash
uv run python scripts/bias_investigation/test_14_channel_audit.py
```

Builds `SimulationDetectionProbability` from `simulations/injections/` at
SNR≥20; queries 1D and 2D quality flags + interpolators at h ∈ {0.70, 0.73,
0.75, 0.77}; computes per-h c_0, unanchored 1D `p̂(c_0)` (M-marginal of 2D
quality flags), anchored 1D values at d_L ∈ {0, 0.05, c_0}, anchored 1D
window mean on [0, c_0]; per-M 2D values weighted by injection density
(reliable mask n_total ≥ 10); analytical cluster-scale projection using
known cluster c_0=0.10, p̂_cluster(c_0)=0.544 from Phase 45 T10.

Output: `scripts/bias_investigation/outputs/phase45/channel_audit.json`.

Pre-registered gates G2a (1D under-anchors) / G2b (1D over-anchors) /
G2c (inconclusive |Δlift| ≤ 0.05).

**Result (2026-05-01):** G2b on cluster-projection — 1D anchored window mean
on [0, 0.10]=0.834 vs 2D unanchored=0.568; Δ=−0.27, sensitivity-robust
across p̂(bin1) ∈ [0.40, 0.55]. Caveat: local data has dl_max ≈ 2.77 Gpc
(c_0=0.025) vs cluster ≈12 Gpc (c_0=0.10), so direct local measurement is
geometrically incompatible; the cluster-projection numbers are analytical
from known cluster bin geometry.

### 6.6 Audit A4 — per-event diagnostic (412 cluster events)

```bash
# After rsyncing cluster diagnostic CSV (already in repo at simulations/cluster_run_phase45_20260501/):
uv run python scripts/bias_investigation/test_16_per_event_diagnostic.py
```

Reads `simulations/cluster_run_phase45_20260501/event_likelihoods.csv`
(append-mode CSV from multiple cluster eval runs; deduplicated to latest
per (event, h)) and `prepared_cramer_rao_bounds.csv`. Reconstructs cluster
MAP, decomposes into Σ log L_i(h) vs −N log D(h), identifies bias-driving
events, computes channel-divergence drivers.

Output: `scripts/bias_investigation/outputs/phase45/per_event_diagnostic.json`.

**Result (2026-05-02):** G4b — bias is broad (top-10 events contribute
21.6% of |pull|, all in first bin d_L < 0.20 Gpc). Decomposition: Σ log L
peaks at h=0.7400, D(h) shifts MAP to 0.7550 (+0.015, 2.7× larger than
per-event L pull, opposite sign). **Anchor escalation targets wrong layer.**

### 6.7 Audit A7 — Closure test at h_true=0.65 *(lean rescaling approach)*

```bash
# 1. Rescale the existing cluster CRB to h_true=0.65
uv run python scripts/bias_investigation/test_17_rescale_crb_to_h065.py

# 2. Re-draw best-guess parameters with fresh seed
uv run python scripts/prepare_detections.py \
    --workdir simulations/closure_h065 --seed 201 --force

# 3. Push to cluster RUN_DIR and submit (dev_cpu_il QOS: max --array=0-3)
ssh bwunicluster 'mkdir -p /pfs/work9/.../run_closure_h065_20260503/simulations/logs
ln -sfn /pfs/.../run_phase45_20260501/simulations/injections \
    /pfs/.../run_closure_h065_20260503/simulations/injections'
rsync -avz simulations/closure_h065/simulations/{,prepared_}cramer_rao_bounds.csv \
    bwunicluster:/pfs/.../run_closure_h065_20260503/simulations/
scp cluster/evaluate_closure_h065.sbatch bwunicluster:~/MasterThesisCode/cluster/
ssh bwunicluster 'cd ~/MasterThesisCode && sbatch --array=0-3 \
    --export=ALL,RUN_DIR=/pfs/.../run_closure_h065_20260503,PROJECT_ROOT=$HOME/MasterThesisCode \
    cluster/evaluate_closure_h065.sbatch'

# 4. Pull posteriors and analyze
rsync -avz bwunicluster:/pfs/.../run_closure_h065_20260503/simulations/posteriors* \
    simulations/cluster_run_closure_h065_20260503/
uv run python scripts/bias_investigation/test_18_closure_analyze.py
```

Output: `scripts/bias_investigation/outputs/phase45/closure_h065.json`.

**Result (2026-05-03):** G7a — pipeline UNBIASED. 1D continuous MAP=0.6517
(bias=+0.0017), 2D=0.6501 (bias=+0.0001). The +0.025 residual at h_true=0.73
is NOT structural; specific to that realization.


---

## 7. Glossary

| Symbol / term | Definition |
|---|---|
| **h** | Dimensionless Hubble parameter, h = H_0 / (100 km/s/Mpc) |
| **h_true** | Injected truth value used in the simulation campaign; production = 0.73 |
| **h_inj** | h-value at which the *injection campaign* draws events (variable across grid) |
| **H** (constant) | `master_thesis_code/constants.py:25` — production simulation truth = 0.73 |
| **TRUE_HUBBLE_CONSTANT** | `master_thesis_code/constants.py:26` — DEAD CODE = 0.7. Only flows into `GalaxyCatalog` (synthetic, Pipeline A removed). Latent footgun (§4.3). |
| **L_cat** | Catalog-term per-event likelihood — sum over cataloged hosts, Gray Eqs. 24–25 |
| **L_comp** | Completion-term per-event likelihood — integral over uncataloged hosts, Gray Eq. 9 |
| **f_i** | Per-event GLADE completeness, B-band luminosity comparison |
| **D(h)** | Selection-function denominator `∫ P_det(d_L(z, h)) · dV_c/dz dz` (Gray Eq. A.19) |
| **p̂(c_0)** | Histogram-derived first-bin estimate of P_det at d_L = c_0 |
| **c_0** | First-bin midpoint of P_det grid; c_0 = dl_max(h)/120 ≈ 0.10 Gpc at h=0.73 |
| **1D / 2D channel** | `_build_grid_1d` (P_det vs d_L) / `_build_grid_2d` (P_det vs d_L, M) interpolators |
| **without_bh_mass** | 1D channel posterior, unconditional on BH mass M |
| **with_bh_mass** | 2D channel posterior, conditional on observed M from CRB |
| **SNR≥20** | Detection threshold (`SNR_THRESHOLD = 20.0` in `constants.py`) |
| **MAP** | Maximum a-posteriori estimate, argmax of posterior over h-grid |
| **σ_boot** | Bootstrap standard deviation of MAP, B=1000 resamples (T08) |
| **GLADE** | Galaxy List for the Advanced Detector Era — galaxy catalog used for catalog term |
| **EMRI** | Extreme Mass Ratio Inspiral — primary signal class |
| **BallTree** | Spatial index over GLADE galaxies for `O(log N)` host queries |

---

## 8. Key Artifacts

| Artifact | Location |
|---|---|
| Diagnostic test scripts (T01–T12) | `scripts/bias_investigation/` |
| Phase 45 diagnostic outputs (T8–T11 JSONs) | `scripts/bias_investigation/outputs/phase45/` |
| Phase 32 D(h) fix validation | `.gpd/phases/32-completion-term-fix/validation/` |
| Phase 43 verification report | `.gpd/phases/43-posterior-calibration-fix/43-VERIFICATION.md` |
| Phase 44 root-cause analysis | `.gpd/debug/resolved/map-0p86-lcat-explosion.md` |
| Phase 45 diagnosis lock-in | `.gpd/HANDOFF-phase45-diagnosis.md` |
| Phase 45 plan-by-plan SUMMARYs | `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-0{1..5}-SUMMARY.md` |
| Cluster posteriors (current head — Plan 45-04 baseline) | `results/phase45_v2_posteriors/`, `results/phase45_v2_posteriors_with_bh_mass/` |
| Cluster posteriors (Plan 45-06 = anchor 0.8873) | `simulations/cluster_run_phase45_option_A/combined_posterior_{no_bh,with_bh}.json` |
| **v2.3 A1 fine-grid script + output** | `scripts/bias_investigation/test_13_fine_grid_map.py`, `outputs/phase45/fine_grid_map.json` |
| **v2.3 A2 channel audit script + output** | `scripts/bias_investigation/test_14_channel_audit.py`, `outputs/phase45/channel_audit.json` |
| **v2.3 A4 per-event diagnostic script + output** | `scripts/bias_investigation/test_16_per_event_diagnostic.py`, `outputs/phase45/per_event_diagnostic.json` |
| **v2.3 A7 rescaling script** | `scripts/bias_investigation/test_17_rescale_crb_to_h065.py` |
| **v2.3 A7 (initial 4-h) closure analysis script + output** | `scripts/bias_investigation/test_18_closure_analyze.py`, `outputs/phase45/closure_h065.json` (verdict withdrawn — see A7-redux) |
| **v2.3 closure-test cluster sbatch (initial)** | `cluster/evaluate_closure_h065.sbatch` |
| **v2.3 A7-redux fine-grid sbatch (2026-05-04)** | `cluster/evaluate_closure_h065_finegrid.sbatch` (11 h-values, 4-task chained array) |
| **v2.3 A7-redux fine-grid analyzer + output** | `scripts/bias_investigation/test_20_closure_finegrid_analyze.py`, `outputs/phase45/closure_h065_finegrid.json` |
| **v2.3 Tier 2 bootstrap-subsample script + output** | `scripts/bias_investigation/test_19_bootstrap_subsample.py`, `outputs/phase45/bootstrap_subsample.json` |
| **v2.3 cluster CSVs (rsynced from cluster)** | `simulations/cluster_run_phase45_20260501/` (Plan 45-04 cramer_rao_bounds, prepared_cramer_rao_bounds, diagnostics/event_likelihoods, fisher_quality), `simulations/cluster_run_closure_h065_20260503/` (initial 4-h closure posteriors), `simulations/cluster_run_closure_h065_20260503_finegrid/` (A7-redux 11-h fine-grid posteriors) |
| Dark siren likelihood derivation | `derivations/dark_siren_likelihood.md` |
| v2.3 audit programme plan | `~/.claude/plans/can-you-critically-think-calm-cocke.md` |
| Resolved debug sessions | `.gpd/debug/resolved/{h0-posterior-residual-bias,map-0p86-lcat-explosion}.md` |

---

## 9. References

- Gray, R. et al. (2020). Cosmological inference using gravitational wave standard
  sirens: A mock data challenge. *Phys. Rev. D* **101**, 122001. arXiv:1908.06050.
  - Eq. 9 (completeness combination), Eqs. 24–25 (catalog L_cat formula),
    Eq. A.19 (selection-function correction).
- Babak, S. et al. (2023). LISA sensitivity and SNR calculations. arXiv:2303.15929.
  Eq. 17 (galactic confusion noise).
- Vallisneri, M. (2008). Use and abuse of the Fisher information matrix.
  *Phys. Rev. D* **77**, 042001. arXiv:gr-qc/0703086.
- Chen, H.-Y. et al. (2018). A two percent Hubble constant measurement from
  standard sirens within five years. *Nature* **562**, 545-547. arXiv:1709.08079.
- Hogg, D. W. (1999). Distance measures in cosmology. arXiv:astro-ph/9905116.
  Eq. 27 (comoving volume element).

---
---

# Appendix A — Chronological Phase Log

The original date-stamped narrative. Each section links back to its catalog
entry in §3 for mechanism-centric reading.

## Phase 9 — Galactic Confusion Noise (2026-03-29)

→ See [Catalog §3.12](#312-galactic-confusion-noise-in-psd-phase-9).

**Commit:** `3bed9fc` — `[PHYSICS] feat(09-01): add galactic confusion noise to LISA A/E-channel PSD`

- **Issue:** LISA PSD was missing galactic foreground noise, making SNR estimates
  unrealistically high and distorting Fisher-matrix parameter uncertainties.
- **Fix:** Implemented `_confusion_noise()` in `LisaTdiConfiguration`, following
  Babak et al. (2023) arXiv:2303.15929 Eq. (17) with observation-time-dependent
  knee frequency.
- **Impact on bias:** Indirect — more realistic Fisher bounds feed into detection
  probability and per-event uncertainties.

---

## Phase 10 — Five-Point Stencil Derivatives (2026-03-29)

→ See [Catalog §3.11](#311-five-point-fisher-derivatives-phase-10).

**Commit:** `a87eeab` — `[PHYSICS] feat(10-01): wire five-point stencil into Fisher matrix`

- **Issue:** Fisher matrix used O(epsilon) forward difference, producing less accurate
  Cramer-Rao bounds on all 14 EMRI parameters including d_L.
- **Fix:** Wired `five_point_stencil_derivative()` as default in
  `compute_fisher_information_matrix()` (`use_five_point_stencil=True`).
  O(epsilon^4) central difference with coefficients (-1, 8, -8, 1) / 12epsilon.
- **Impact on bias:** More accurate Fisher bounds improve d_L and sky-localization
  uncertainties, tightening the per-event GW likelihood.
- **Reference:** Vallisneri (2008) arXiv:gr-qc/0703086.

---

## Phase 11.1 — KDE to IS Detection Probability (2026-03-31)

→ See [Catalog §3.10](#310-kde--simulation-based-is-p_det-estimator-phase-111).

**Commits:**
- `e86e826` — replace KDE `DetectionProbability` with `SimulationDetectionProbability`
- `caf8ce6` — delete old KDE class and clean up references

- **Issue:** KDE-based P_det had bandwidth sensitivity, poor tail coverage, and
  systematic over/underestimation at grid boundaries.
- **Fix:** Replaced with simulation-based importance sampling (IS) estimator using
  injection campaign data. `SimulationDetectionProbability` builds a histogram
  grid from injected events with proper IS weights, then interpolates via
  `RegularGridInterpolator`.
- **Impact on bias:** Removed KDE artifacts from P_det surface. Variance reduction
  factor 11.8-24.9x in boundary bins. Backward-compatible API.

---

## Gaussian Index Bug Investigation (2026-03-31)

→ Eliminated hypothesis (see [§5](#5-eliminated-hypotheses)).

- **Issue:** Investigated whether `bayesian_statistics.py` lines 631/831 used
  Gaussian index [0] (3D, without BH mass) instead of [1] (4D, with BH mass)
  in the "with BH mass" numerator.
- **Finding:** Under delta-function approximation, M_frac ~ 1 with sigma ~ 1e-7,
  so 3D vs 4D Gaussian makes no numerical difference. **Not a root cause.**
- **Status:** Eliminated as hypothesis.

---

## Phases 14-15 — Likelihood Derivation and /(1+z) Fix (2026-03-31)

→ See [Catalog §3.7](#37-spurious-1z-jacobian-phase-15--minor-insufficient-alone).

**Commits:**
- `c466e86` — derive d_L-only dark siren likelihood (Phase 14)
- `1d4e9a1` — remove spurious /(1+z) Jacobian from with-BH-mass numerator (Phase 15)
- `46e2662` — validation: /(1+z) fix insufficient for bias

- **Phase 14:** First-principles derivation of dark siren likelihood from Bayes'
  theorem. Verified sky-localization weight placement, dimensional consistency,
  and mapped all 12 terms to code. Deliverable: `derivations/dark_siren_likelihood.md`.
- **Phase 15:** Code audit found spurious `/(1+z)` factor at `bayesian_statistics.py`
  line 646 (and 871). Removed per Jacobian absorption identity.
- **Impact on bias:** The /(1+z) factor was a code anomaly but **not the dominant
  bias mechanism**. Posterior remained monotonically decreasing after fix.
- **Conclusion:** Necessary correction but insufficient for bias resolution.

---

## Phases 17-20 — Injection Campaign Validation (2026-03-31 to 2026-04-01)

→ Eliminated hypothesis (see [§5](#5-eliminated-hypotheses)).

**Key commits:**
- `60fe875` — characterize waveform failure modes in injection campaign
- `6a1ac4d` — add quality flags to `SimulationDetectionProbability`
- `f15df43` — IS-weighted histogram estimator for P_det grid
- `74affb4` — P_det validation framework

- **Phase 17:** Injection physics audit — confirmed 14-parameter consistency,
  d_L round-trip accuracy to 2e-13 precision.
- **Phase 18:** Grid quality assessment — Wilson confidence intervals, detection
  yield analysis, quality flags.
- **Phase 19:** IS estimator construction with proper weights from 463k injections.
- **Phase 20:** Validation framework — VALD-01 PASS (916 bins), VALD-02 PASS
  (alpha_grid = alpha_MC exactly).
- **Conclusion:** P_det surface is validated as **not the bias source**.

---

## Phases 21-23 — Posterior Numerical Stability (2026-04-02)

→ Eliminated hypothesis (see [§5](#5-eliminated-hypotheses)).

**Commits:**
- `7df0baa` — implement posterior combination module with 4 strategies
- `db5eb2b` — implement physics-floor strategy with per-event min-nonzero floor

- **Issue:** Float64 underflow in product of 500+ per-event likelihoods. Some events
  have p(d_i|h) = 0.0 at certain h-bins (no catalog host found), causing
  zero-product collapse of the joint posterior.
- **Fix:** Log-space accumulation (`posterior_combination.py`) with four strategies:
  log-sum, per-event floor, per-event nonzero-min, global floor. Physics-motivated
  floor from faintest catalog galaxy at error volume boundary.
- **Impact on bias:** Eliminated numerical artifacts but did not resolve the
  systematic catalog-driven bias.

---

## Phases 24-25 — Completeness Correction (2026-04-04)

**Commits:**
- `2341b80` — refactor `GladeCatalogCompleteness` with f(z, h) interface
- `f60a75a` — completeness-corrected dark siren likelihood (Gray et al. 2020 Eq. 9)

- **Issue:** GLADE catalog reaches only ~21% completeness at typical EMRI distances
  (>796 Mpc, z > 0.08). The asymmetric galaxy density distribution systematically
  biases the posterior toward lower H0: at trial h < h_true, galaxies at z < z_true
  can match the observed d_L, and there are more galaxies at lower z due to
  catalog incompleteness.
- **Fix:** Implemented completeness-corrected likelihood per Gray et al. (2020)
  arXiv:1908.06050 Eq. 9:

  ```
  p_i(H0) = f(z, H0) * L_cat + (1 - f(z, H0)) * L_comp
  ```

  where L_cat is the catalog term (sum over cataloged galaxies) and L_comp is the
  completion term integrating over uncataloged hosts weighted by a comoving volume
  prior. f(z, H0) from GLADE+ B-band luminosity comparison.
- **Impact on bias:** Primary mitigation for catalog-driven bias. However, the
  completion term itself carried systematic bias (later resolved by Phase 32).

---

## P_det Grid Extrapolation Fix (2026-04-08)

→ See [Catalog §3.9](#39-p_det-extrapolation-fill_value00-v14--commit-44d5358).

**Commit:** `44d5358` — `[PHYSICS] fix P_det grid extrapolation causing 44% completeness fallback`

- **Issue:** `RegularGridInterpolator` used `fill_value=0.0` for out-of-grid queries.
  702 events received P_det = 0, causing the completeness correction to fall back
  to catalog-only likelihood (L_comp contribution = 0).
- **Fix:** Changed `fill_value` from `0.0` to `None` (nearest-neighbor extrapolation).
- **Impact on bias:** Eliminated 702 zero-likelihood completeness fallbacks. Improved
  MAP h from 0.663 to 0.680 (+0.017). Reduced bias from -9.2% to -6.9%.

---

## Phase 32 — Full-Volume D(h) Denominator Fix (2026-04-08, v1.5/v2.1 transition)

→ See [Catalog §3.1](#31-local-window-l_comp-denominator-phase-32--map-013-local).

**Commit:** validated in `.gpd/phases/32-completion-term-fix/validation/` (D(h)
fix later integrated into the production combine path via Phase 43 commit `2853c32`).

- **Issue:** `L_comp` denominator restricted to the local 4-σ d_L window, while Gray
  et al. (2020) Eq. A.19 normalizes over the full detectable volume. Local-window
  normalization fails to cancel h-dependent volume effects, producing a U-shaped
  L_comp(h) anti-correlation with the truth and amplifying the catalog-incompleteness
  bias.
- **Fix:** Replaced local-window denominator with the precomputed full-volume
  `D(h) = ∫ P_det(d_L(z, h)) · dV_c/dz dz` table, evaluated once per h and reused for
  every event. New helper: `precompute_completion_denominator()`.
- **Impact on bias (local 59-event SNR≥20 dataset):** MAP shifted 0.60 → 0.73 for
  both channels — bias 0.0% (pre-fix -17.8%). L_comp(h) now monotonically increasing
  for all sampled events; 0/1593 NaN/zero values.
- **Reference:** Gray et al. (2020) arXiv:1908.06050 Eq. A.19. Full validation:
  `.gpd/phases/32-completion-term-fix/32-02-SUMMARY.md`.

---

## v2.1 H0 Bias Resolution Milestone (Shipped 2026-04-09)

**Phases 30–34, GSD-tracked.** Milestone closed with bias eliminated on the local
SNR≥20 dataset; cluster production validation deferred to v2.2.

| Phase | Topic | Outcome |
|-------|-------|---------|
| 30 | Evaluation infrastructure | `BaselineSnapshot`, `extract_baseline()`, `--save_baseline`/`--compare_baseline` CLI |
| 31 | Catalog-only diagnostic | `--catalog_only` flag + per-event L_cat/L_comp/f_i CSV; **confirmed L_comp as dominant bias source** |
| 32 (GPD) | Full-volume D(h) fix | MAP 0.60 → 0.73 locally (Eq. A.19) — see entry above |
| 33 | P_det grid resolution | 38-point cluster sweep with N_bins ∈ {30, 60} showed zero MAP delta — **grid resolution not a bias source** (Audit A8 reservation: re-check vs current baseline) |
| 34 | Fisher matrix quality | `allow_singular=True` removed, condition-number gate, `fisher_quality.csv`, two-panel `plot_fisher_diagnostics()` figure |

Milestone artifacts: `.planning/milestones/v2.1-biasres-ROADMAP.md`.

---

## v2.2 Pipeline Correctness Milestone (2026-04-21 — 2026-04-27, CLOSED)

After the v2.1 fixes, the cluster production posterior persisted at MAP≈0.86 — a
much larger discrepancy than the local SNR≥20 result suggested. v2.2 audited the
full simulation+evaluation pipeline. It delivered three independent fixes
(coordinate frame, parameter-estimation, statistical correctness) and a
verification gate that uncovered the final calibration bug.

### Phase 35 — Coordinate Bug Characterization (2026-04-21)

- Audited the existing equatorial→ecliptic mismatch suspected from the v2.1 production
  results. Confirmed that GLADE host coordinates were ingested in equatorial RA/Dec
  while EMRI sky angles `qS, phiS` were defined in ecliptic — a frame mismatch of up
  to the obliquity 23.4°, while the typical BallTree search radius is only ~1.8°.
- Locked a regression pickle `36-superset-regression.pkl` as the Phase 40 VERIFY-02
  anchor.

### Phase 36 — Coordinate Frame Fix (2026-04-22)

→ See [Catalog §3.2](#32-equatorial-crbs-vs-ecliptic-glade-phase-36--phase-43-h2--host-recovery-313860).

**Commits:** `b460297` (COORD-03), `c17ecb6` (COORD-02), `b2ef9c9` (COORD-04),
`5b9cfbf` (COORD-02b).

- **COORD-03:** Equatorial→ecliptic rotation on GLADE ingestion via
  `astropy.coordinates.BarycentricTrueEcliptic(J2000)`.
- **COORD-02:** Polar-correct BallTree Cartesian embedding `(sin θ cos φ, sin θ sin φ,
  cos θ)`.
- **COORD-04:** Eigenvalue sky search radius from the 2×2 Fisher sky covariance with
  `|sin θ|` Jacobian.
- **COORD-02b:** 4D BallTree sky sub-space uses spherical embedding.
- **Impact on bias:** Necessary precondition for correct host matching, but
  insufficient alone — the production posterior remained at MAP≈0.86. The angular
  fix was confirmed required (obliquity 23.4° ≫ 1.76° median BallTree radius) but
  surfaced a separate bug in the CRB CSV that Phase 43 later resolved.

### Phase 37 — Parameter Estimation Correctness (2026-04-23)

→ See [Catalog §3.8](#38-per-parameter-fisher-epsilon-phase-37-pe-02).

**Commits:** `55a6d99` (PE-01), `7429c6e`/`16ce20f` (PE-02).

- **PE-01:** Threaded `h_inj` into `set_host_galaxy_parameters` so injection-time
  distance scaling matches the target Hubble constant per event.
- **PE-02:** Per-parameter `derivative_epsilon` for all 14 EMRI parameters (replacing
  a single global epsilon that was poorly tuned for the slow parameters).

### Phase 38 — Statistical Correctness (2026-04-23)

→ See [Catalog §3.6](#36-l_cat-formula-phase-38--gray-eqs-2425-normalization).

**Commits:** `005e792` (L_cat fix), `a70d1a2` (symmetric zero-fill).

- **L_cat formula:** Replaced ad-hoc per-galaxy normalization with
  `L_cat = (1/N) · Σ_g (N_g / D_g)` per Gray et al. (2020) Eqs. 24–25.
- **Symmetric P_det zero-fill** + an off-grid quadrature diagnostic (an early
  warning surface for the bug Phase 44 later root-caused).

### Phase 39 — HPC & Visualization Safe Wins (2026-04-24)

Pressure-gated GPU memory management, batched CRB writes (flush_interval 1 → 25),
LaTeX auto-detection in figures, bootstrap HDI band on convergence plots.
Software-only — no physics impact, but removed friction from cluster runs.

### Phase 40 — Verification Gate (2026-04-23 — 2026-04-25)

Six-plan retrospective verification of v2.2 (VERIFY-01..VERIFY-05).

- VERIFY-01 PASS — 544 tests green, D-06 inventory complete.
- VERIFY-02 PASS — abort-gate diagnostic shows 0% MAP shift on the regression anchor.
- **VERIFY-03 FAIL** — h-sweep reproduced **MAP = 0.860** (expected 0.73 ± 0.01).
  Root cause not yet localized; this triggered Phase 43.
- VERIFY-04 STAGE-2-TRIGGER — Q3 quartile anisotropy `|Δ/σ| = 5.4` (later resolved
  by Phase 43 H2 fix; Phase 42 deferred).
- VERIFY-05 BORDERLINE — `mean_lb = 0.041`; Phase 41 SKIPPED per user decision.
- VERIFY-06 plan summary closed with `GAPS_FOUND` and triggered Phase 43.

---

## Phase 43 — Posterior Calibration Fix (2026-04-26 — 2026-04-27)

→ See [Catalog §3.2](#32-equatorial-crbs-vs-ecliptic-glade-phase-36--phase-43-h2--host-recovery-313860) (H2)
and [§3.3](#33-extract_baseline--combine-missing--n-log-dh-phase-43-h1--removes-monotone-h-pull) (H1).

**Commits:** `2853c32` (D(h) added to combine path), `ab4bc80` (Fisher covariance
ecliptic rotation), `a2df67b` (extract_baseline deprecation + CRB ecliptic migration),
`261091a` (phase complete: MAP=0.730).

- **Issue:** Phase 40 VERIFY-03 SC-3 surfaced `MAP = 0.860` from the production
  `extract_baseline()` h-sweep on the v2.2 baseline. Two competing root causes:
  - **H1:** `combine_log_space` / `extract_baseline` accumulate `Σ log L_i(h)` but
    omit the `−N · log D(h)` selection-function correction from Gray Eq. A.19. Without
    it, the posterior is biased toward `argmax Σ log L_i`, which monotonically grows
    with h.
  - **H2:** CRBs on disk (`prepared_cramer_rao_bounds.csv`) still stored equatorial
    sky angles `qS, phiS` even after the Phase 36 GLADE rotation — a residual
    coordinate mismatch only on the disk-cached CRBs.
- **Fix:**
  - **H1:** Added the Gray Eq. A.19 D(h) selection-function correction to the
    production combine path (`2853c32`); deprecated `extract_baseline` with a logged
    warning that documents the missing normalization (`a2df67b`).
  - **H2:** Rotated the 14×14 Fisher covariance CRBs to ecliptic via the
    `migrate_crb_to_ecliptic` script (`ab4bc80`); added `_coord_frame =
    ecliptic_BarycentricTrue_J2000` provenance tag.
- **Result:** Post-fix `--evaluate` MAP **0.730** ∈ [0.72, 0.74] — VERIFY-03 SC-3
  PASS. Host recovery 31/60 → 38/60; "no possible hosts" 10 → 1; Phase 40 Q3
  anisotropy 5.4σ → 0σ; Phase 42 DEFERRED. Both root causes confirmed independently
  by the Phase 43 verifier (computational oracle 8/8 PASS) but not yet by ablation
  (Audit A5).
- **Reference:** `.gpd/phases/43-posterior-calibration-fix/43-VERIFICATION.md`.

---

## Phase 44 — h-Dependent P_det Zero-Fill Cutoff (2026-04-28 — 2026-04-29)

→ See [Catalog §3.4](#34-h-dependent-p_det-zero-fill-cutoff-at-c_0--1h-phase-44--cluster-0860--07650).

**Commit:** `3697bdd` — `[PHYSICS] Fix h-dependent P_det zero-fill cutoff at c_0 ∝ 1/h`.

- **Issue:** The cluster posterior persisted at MAP=0.860 even after Phase 43's H1
  fix. Debug session `.gpd/debug/resolved/map-0p86-lcat-explosion.md` traced the
  pathology to `detection_probability_without_bh_mass_interpolated_zero_fill`:
  a left-side cutoff at `dl_centers[0] = dl_max(h)/120` returned `p_det = 0` for
  any source below that threshold. Because `dl_max(h) ∝ 1/h`, the cutoff
  `c_0(h) ∝ 1/h` was a *moving* threshold:
  - `c_0(0.73) = 0.0998 Gpc`, `c_0(0.86) = 0.0847 Gpc`.
  - 4 close events with `d_L ∈ [0.085, 0.097] Gpc` had `p_det = 0` at `h = 0.73`
    but `p_det ≈ 0.55` at `h = 0.86`. Their `L_comp` therefore "switched on" only
    at high h, contributing **+145.7 log-units** toward `h = 0.86` across 312
    events — pinning the cluster MAP at 0.860.
  - The threshold was a *bin-midpoint artifact*, not the injection minimum: the
    first bin spans `[0, 2 c_0)` and is densely populated by GLADE-low-z
    injections (`n_total[0] = 312`).
- **Fix:** Removed the left-side zero-fill so the existing
  `RegularGridInterpolator(fill_value=None)` returns the genuine first-bin estimate
  `p̂(c_0) ∈ [0.47, 0.60]` for `d_L < c_0`. Right-side zero-fill kept (sources beyond
  the injection horizon are genuinely undetectable).
- **Impact on bias:** Cluster re-eval (jobs 4160638/4160639) on production seed200
  (412 events, SNR≥20): **MAP shifted 0.860 → 0.7650**. The +145.7 log-unit
  pathology is gone; all 4 zero-handling strategies now produce identical MAPs
  (no events suppressed). 68% equal-tailed interval [0.750, 0.765] — does not yet
  contain the truth.
- **Residual:** +0.035 (3 σ_boot) above truth — not statistical (bootstrap σ ≈
  0.011); deferred to Phase 45.

---

## Phase 45 — P_det First-Bin Asymptote Fix (Active, ESCALATING, 2026-04-30 — present)

→ See [Catalog §3.5](#35-p_det-first-bin-asymptote-underestimate-at-d_l--c_0-phase-45-active).

**Commits:** `09ee262` (Plan 45-02 single anchor), `4a260e9` (Plan 45-04 hybrid).
Side branches `phase-45-option-A` (`ca64ffb`) and `phase-45-option-D` (`12d3ebc`)
pre-stage further escalations — **DEPLOYMENT BLOCKED** pending v2.3 audits.

- **Issue:** After Phase 44, the cluster MAP residual `+0.035` is *systematic*
  (bootstrap B=1000 gives 68% interval [0.745, 0.765] excluding truth 0.73 by
  ≈3 σ_boot=0.0114). Diagnostic battery (T8–T11):
  - **T8 (bootstrap):** systematic, σ_boot=0.0114.
  - **T9 (first-bin density):** upper-skew ratio 3.22 (29 events in upper third
    vs 9 in lower third of `[0, 2 c_0]`); weighted mean `d_L = 0.132 Gpc` (above
    midpoint 0.10).
  - **T10 (empirical asymptote):** 16/16 detected for `d_L < 0.10 Gpc`,
    `p̂ = 1.000` Wilson 95% LB 0.806; interpolator returns 0.544 at `c_0` and
    0.748 at `d_L → 0` (linear extrapolation through bins 0,1) — a 0.25–0.46
    underestimate vs the empirical truth.
  - **T11 (window proximity):** 0/60 events touch `d_L = 0` but **26/60 (43%)**
    integrate across `c_0`. Anchor-at-zero alone is largely inert; the lift must
    cover `[0, c_0]`.
- **Mechanism:** `p̂(c_0) ≈ 0.544` is biased downward because the upper-skewed
  injection density makes the histogram mean dominated by the high-`d_L`
  (low-`p_det`) end. Linear extrapolation partly recovers the trend but still
  systematically *underestimates* `p_det` in `[0, c_0]`, suppressing `L_comp` at
  low h and biasing MAP upward.
- **Plan 45-01 (h-independence)** — `49be6c0`. Replaced an ill-posed
  scalar-spread gate with a likelihood-ratio (G-test) test of binomial-rate
  homogeneity. Result: G = 7.30, dof = 5, p = 0.199 — cannot reject h-homogeneity
  across all `h_inj` injection groups. Recommended anchor:
  `_P_MAX_EMPIRICAL_ANCHOR = 0.7931` (pooled Wilson 95% LB from n=71, k=63).
- **Plan 45-02 (single anchor)** — `09ee262`. Prepended `(0.0, 0.7931)` to
  `(dl_centers, p_det_1d)` in `_build_grid_1d`. 564 CPU tests pass; 7 new
  `TestPhase45EmpiricalAnchor` regression tests; docstring corrected
  (`linear extrapolation` not `nearest-neighbour`).
- **Plan 45-03 (cluster re-eval)** — `eae4388`/`d2dc28b`. **MAP unchanged at
  0.7650**. Posterior peak height dropped 0.373 → 0.347 with mass redistributed
  to bins 0.745–0.755 — a continuous shift below the discrete grid step
  Δh=0.005. **Branch B (UNDER-CORRECTION) — escalate.**
- **Plan 45-04 (hybrid 4c)** — `4a260e9`. Added a second anchor
  `(0.05, 1.0)` to lift `[0, c_0]` more aggressively. Pre/post probes confirm
  `interp(0.05; h=0.73)` lifted from 0.6687 to 1.0 (+0.331); window-average lift
  ≈ +0.166 over `[0, c_0]`. 509 CPU tests pass. h-spread at d_L=0.05 is exactly
  0 (intermediate is a fixed scalar at fixed physical position).
- **Plan 45-05 (cluster re-eval)** — `962fce3`. **MAP shifted 0.7650 → 0.7550**
  (one discrete grid step toward truth); bootstrap σ_MAP tightened
  0.0117 → 0.0099. 68% interval still [0.7450, 0.7650] — truth h=0.73 still
  outside by 0.015. **Branch B (UNDER-CORRECTION, improved) — escalate again.**
  - `without_bh_mass`: MAP=0.7550 (Δ-truth = +0.025, ≈2.5 σ_boot).
  - `with_bh_mass`: MAP=0.7450 (unchanged from 45-03 because `_build_grid_2d` was
    not patched).
  - Apparent saturation: continuous Δ MAP per round ≈ −0.005 even though the
    local lift jumped from +0.04 (45-02) to +0.166 (45-04). Hypothesized causes
    (Audits A2/A3/A4 will pin down):
    1. L_comp gradient saturation,
    2. Window-proximity skew (event windows centroid away from d_L=0.05),
    3. `_build_grid_2d` (with_bh_mass channel) being the larger remaining lever.
- **Plan 45-06 (Option A, pre-staged on `phase-45-option-A` branch)** —
  `ca64ffb`. Raises the d_L=0 anchor from the conservative Wilson LB 0.7931 to
  the point estimate 0.8873. Single-line change. **Cluster verification BLOCKED
  pending Audits A4 + A7** (A2 G2b cluster-projection suggests direction may
  be wrong — current 1D over-anchors structurally vs 2D).
- **Plan 45-07 (Option D, pre-staged on `phase-45-option-D` branch)** —
  `12d3ebc`. Extends the hybrid anchor logic to `_build_grid_2d` so the
  with_bh_mass channel sees the same `[0, c_0]` lift. Predicted to move
  with_bh_mass MAP 0.7450 → ≈0.7400. **Cluster verification BLOCKED pending
  Audits A4 + A7** (A2 cluster-projection: 1D anchored mean lift 0.834 vs 2D
  unanchored 0.568 = Δ−0.27; extending the over-correction to 2D risks
  pushing with_bh_mass MAP below truth).

---
---

# Appendix B — Decision History

Rejected approaches and approved-pending-validation decisions, with one-line
justifications. Decisions are logged here so future sessions don't relitigate
options that were already considered.

| Decision | Status | Reasoning |
|---|---|---|
| 30-bin P_det grid (Phase 33) | REJECTED | 38-point cluster sweep with N_bins ∈ {30, 60} showed zero MAP delta — grid resolution not a bias source (CONFIRMED by Audit A8 G8a 2026-05-01: baseline-invariant test holds against post-Phase-43 baseline) |
| Sub-binning of first P_det d_L bin (RESEARCH §4b, Plan 45-04 planning) | REJECTED-AT-PLANNING; FLIPS TO ACCEPTED if Audit A3 = G3b | Rejected as "more code surface"; the v2.3 audit programme tests this rejection by quantitatively pre-predicting the MAP shift before implementation |
| Plan 45-04 Option B (intermediate at d_L=0.025) | REJECTED | Empirically wrong direction (lifts [0, 0.025] high but lowers [0.025, 0.05]; net ~zero on window integral) |
| Plan 45-04 Option C (third anchor at d_L=0.075) | REJECTED | Similar to Option A but more code surface for marginal gain |
| Plan 45-04 Option E (accept current MAP=0.755 / 0.745 as paper result) | DEFERRED | Pending closure-test (Audit A7) outcome; if A7=G7a, this becomes acceptable with documented systematic |
| Plan 45-06 deployment (raise d_L=0 anchor 0.7931 → 0.8873) | DEPLOYMENT BLOCKED | Pending Audits A4 + A7 (A0/A1/A2/A5/A8 done 2026-05-01). A2 G2b cluster-projection finding suggests raising the anchor would widen the lift gap (1D already over-anchors vs 2D by Δ=−0.27); deployment direction not yet signed-correct. Anchor's statistical support (n=71, k=63, Wilson 95% [0.793, 0.942]) doesn't justify 0.005-grid precision target. |
| Plan 45-07 deployment (extend hybrid to `_build_grid_2d`) | DEPLOYMENT BLOCKED | A2 done G2b 2026-05-01: cluster-projection shows 1D over-anchors by Δ=−0.27 vs 2D's natural extrapolation. Extending the same anchor system to 2D would impose this over-correction on the channel currently closer to truth (with_bh_mass MAP=0.7450 → predicted 0.7400, below truth). Pending A4 (per-event evidence) + A7 (closure-test) before deployment. |
| Audit A0 (Hubble-constant consistency check) | DONE — G0a CLEAN (2026-05-01) | `TRUE_HUBBLE_CONSTANT=0.7` is dead-code default in synthetic catalog; production paths use `H=0.73` end-to-end |
| Delete `TRUE_HUBBLE_CONSTANT` and `GalaxyCatalog` (synthetic, Pipeline A) | LOW-PRIORITY CLEANUP | Latent footgun; Pipeline A removed in commit `c1571a2`; benchmark test still uses the class |
| Delete `extract_baseline()` (currently deprecated by warning only) | LOW-PRIORITY CLEANUP | Footgun; future code may call it and get biased MAP with only a log warning |
