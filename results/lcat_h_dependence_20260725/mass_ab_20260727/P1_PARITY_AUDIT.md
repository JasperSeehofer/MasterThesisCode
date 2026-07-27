# P1 probe/production parity audit — FIX-3 §7.1 (mandated by P5's consistency clause)

**Date: 2026-07-28. Status: COMPLETE. Verdict up front: parity is NOT broken
— the ×3–5 "shortfall" is an axis-translation error in the gate arithmetic,
and hypothesis (iv) (m-clamp suppression) is REFUTED in its stated form:
the clamped queries carry ~75–90 % of the measured conditioning movement,
they do not suppress it.** The pre-registered "(d1) reappears at ≈ −6.5 ln
on the clamp-free pool" prediction (RUNBOOK_NEXT_SESSION_4 thread 1) must
NOT be carried by campaign #51 in that form; the replacement prediction is
in §6.

## 0. Setup / provenance

- Scripts + outputs (this audit, all offline, CPU-only, no source edits):
  `p1_parity/p1a_production_sigma.py` → `p1a_results.json`,
  `p1a_gridonly.json` (env `MTC_WBH_GRID_ONLY=1` arm);
  `p1_parity/p1b_probe_and_translation.py` → `p1b_results.json`;
  `p1_parity/p1c_dgen_reconciliation.py` → `p1c_results.json`;
  `p1_parity/p1d_acell_hybrid_translation.py` → `p1d_results.json`
  (the §1 tautology rows and the §2 production-hybrid A-cell rows).
- Production path reproduced by IMPORTING the production modules:
  `SimulationDetectionProbability` built with the exact bs.py:2064–2081
  kwargs (dl_bins 60, mass_bins 40, local_linear, `pdet_z_resolved=True`,
  flag off/on) on the canonical seed-1000 pool
  (`data/injections/injection_h_0p73_task_*.csv`, md5-identical to the A/B
  cells' pool); catalogue rows via
  `GalaxyCatalogueHandler(M_min=10^4.5, M_max=10^6, z_max=1.5)`; the
  Σ_glob_wbh sum assembled exactly as
  `precompute_global_catalog_selection` (with_bh=True, isotropic,
  w_g = R_eff(M_g)/(1+z_g), eligibility z < z_max(h)).
- Catalogue anchors MATCH the probe's z0 snapshot exactly
  (n_pruned 9,060,017; n(z<0.992) 9,060,008; W = 6.34766393939e8 — the
  #40b catalogue regeneration of 2026-07-27 did not change (z, M)).
- **Harness validation (MEASURED):** the local rebuild reproduces the
  cluster cellApp flag-OFF Σ_glob_wbh log values to all 7 printed digits
  at every venue h, and the flag-ON values to ≤1e-7 relative against the
  Σ_on table back-solved from the zmzApp/cellApp per-event
  `L_cat_with_bh` ratio (which is constant across all 876 catalogue
  events at std < 1e-13 — proving the flag's A-cell effect is EXACTLY the
  Σ_glob_wbh swap; `w_G`, `L_cat_no_bh`, `B_num`, `L_comp` are
  bit-identical between A″ and A‴).

**The measured production Σ_glob_wbh tables** (cluster; off = cellApp logs,
on / grid-only = back-solved from zmzApp / zmzGridOnly diagnostics; local
rebuild agrees):

| h | Σ_off | Σ_on (joint shrunk) | Σ_grid-only |
|---|---|---|---|
| 0.73 | 2.847445e8 | 2.920316e8 | 2.989355e8 |
| 0.80 | 2.918739e8 | 3.003153e8 | 3.062085e8 |
| 0.86 | 2.976373e8 | 3.074504e8 | 3.120567e8 |

dln(0.73→0.86): off **0.044283**, on **0.051452**, grid-only **0.042957**;
flag-raw Δdln = **+0.007168**, conditioning-only (on − grid-only) Δdln =
**+0.008495**. Probe (z3, shrunk − mz): **+0.009004** — production/probe
conditioning-slope parity to **6 %**.

## 1. The headline reconciliation — the shortfall is the AXIS, not parity

**The exact A-cell translation.** In `absolute_marginal` the flag scales
`L_cat_with_bh` by Σ_off(h)/Σ_on(h) per event and touches nothing else
(measured, §0). The exact predicted profile shift for any Σ pair is
Σ_i ln(1 + s_i (r−1)) computed from the cellApp per-event diagnostics,
where s_i = w_G·L_cat_wbh/combined_wbh is the event's catalogue-channel
share. Feeding the production Σ tables back through it reproduces the
readout's measured numbers to the last digit (tautology check — the loop
closes at machine precision):

| Σ pair (through the exact translation) | Δ(2D)@0.80 | Δ(2D)@0.86 |
|---|---|---|
| production on/off (raw) | **−0.513** | **−1.181** | (readout: −0.51 / −1.18)
| production grid-only/off | +0.560 | +1.008 | (readout: +0.56 / +1.01)
| production on/grid-only (conditioning-only) | **−1.081** | **−2.202** | (readout: −1.07 / −2.19)
| **probe shrunk/mz (the probe table on the correct axis)** | **−1.155** | **−2.346** | (vs measured conditioning-only −1.08 / −2.20: **7 % agreement**)

**The multiplier mismatch (MEASURED).** The A-cell's effective multiplier
on a Δdln Σ movement is Σ_i s_i = **232.7 / 224.3 / 217.8** at h =
0.73/0.80/0.86. The z3 −6.5 gate lives on the D_gen (generator/B) axis,
whose implied multiplier is |−6.4695| / 0.009004 = **718.5**. Ratio:
**×3.2** — exactly the observed "×3–5 shortfall". First-order check:
−(≈225)·0.008495 = −1.9 ≈ the measured conditioning-only −2.19 (the exact
per-event form above closes the remainder). **The probe table, translated
through the correct axis, predicts the measured A/B result.** P3's
"[−9, −3] = the shipped −6.5 increment re-expressed at 0.80" carried the
D_gen-axis multiplier onto the A-cell axis without re-deriving it; the
NULL verdict at gate level was an artifact of that transfer, while the
physics (table movement + translation) is fully consistent.

## 2. Hypothesis (iv), the m-clamp — REFUTED as stated, mechanism inverted

- **Clamp fraction on the ACTUAL query set (h = 0.73, MEASURED):**
  **84.4 %** of the eligible rate weight w_g queries at
  log₁₀M_z ≥ m_max(pool) = 6.000 (row-level; the readout's 81.4 % is the
  binned-profile version of the same number — the difference is the
  0.113-wide lm bins straddling the threshold); those clamped queries
  contribute **68.5 %** of Σ_off and **69.3 %** of Σ_on.
- **Clamped queries DO feel the u-conditioning.** The claim "clamped
  queries are insensitive to the u axis by construction" is false for the
  shipped table: at the top m-node the (K5)-shrunk survival varies across
  the catalogue's u range by ~20 % at d_L = 2 Gpc (0.242→0.292 across the
  10–90 % catalogue u-quantiles) and ~25 % at 4 Gpc; the top-m-node row is
  NOT starved (ESS median 360, min 18; shrinkage w median 0.973, min 0.648)
  because the pool piles up at the 10⁶ cap.
- **Decomposition of the measured movement (hybrid sums, production
  query set):** switching the conditioning ONLY on unclamped queries gives
  Δdln(0.73→0.86) = +0.000769 (**11 %** of the full +0.007168); ONLY on
  clamped queries gives +0.006424 (**90 %**). Through the exact A-cell
  translation: unclamped-only −0.17 @0.86, clamped-only −1.02 @0.86 (full
  raw −1.18). Probe-side twin (p1b, binned profile): unclamped-only
  −0.54 @0.86, clamped-only −1.79, full −2.35 → clamped share 74–76 %.
  **The clamped weight CARRIES ~75–90 % of the (d1) movement** — via the
  u-dependence of the m_max boundary node — rather than suppressing it.
- **Restricted-increment test (the pre-registered prescription: probe
  increment restricted to the unclamped weight vs the measured −2.19).**
  If clamp suppression explained the shortfall, the unclamped-restricted
  prediction would match −2.19 and the full prediction would overshoot it
  ×3–5. MEASURED: restricted (probe, correct axis) = **−0.54** @0.86 —
  4× too small; FULL (probe, correct axis) = **−2.35** — matches −2.20
  within 7 % (§1). **Clamp suppression quantitatively fails; axis
  translation quantitatively succeeds.**

Consequence for (g1): the clamp remains a genuine SUPPORT limitation — the
movement measured on the clamped 84 % is the u-dependence of a boundary
extrapolation, not of a measured conditional at the catalogue's masses —
but its role in the §7.1 NULL is the opposite of the readout's consequence-2
narrative, and §6 restates the campaign prediction accordingly.

## 3. Item (i) — 2D-channel Σ parity, binned profile vs row-by-row

- Denominators agree: row-by-row pooled no-BH Σ at 0.73 = **5.12243e8**
  (probe binned pooled 5.122e8, ratio ≈ 1.000). The registered
  0.556-vs-0.589 item is therefore entirely a **numerator (with-BH object)
  discrepancy**: production Σ_off 2.847445e8 vs probe binned M_z-only
  3.01822e8 → probe/production = **1.0600** (0.589/0.556 = 1.059 ✓ same
  item).
- Decomposition (p1b, same estimator on both sides) — the 1.0600
  factorises cleanly:
  - **binning: ×1.0103** (probe q_lm on the 300×60 binned profile vs the
    same q_lm on the 9.06M rows: 3.01822e8 / 2.98744e8);
  - **grid/interpolant convention: ×1.0491** (row-by-row probe-grid
    q_lm, 41 uniform-log-M nodes + step-d_L, vs row-by-row production
    `RegularGridInterpolator`, 40 geomspace-M_z × 60 linear-d_L:
    2.98744e8 / 2.847445e8). Product 1.0599 ✓. The production grid-only
    control measured the same class of difference at +4.98 % in value
    for the 31-node/DLQ-3000 joint machinery — consistent.
  - Conditioning-object twin: probe shrunk-joint row-by-row 2.91855e8 vs
    production flag-ON 2.920316e8 → **−0.06 %** (binned: +0.75 %) — the
    joint objects agree essentially exactly; the 6 % item is a property
    of the BASELINE arm's grid conventions, not of the new estimator.
- Slope parity: conditioning-only Δdln(0.73→0.86) probe binned +0.009004
  / probe row-by-row +0.008214 vs production (on − grid-only) +0.008495
  (**3–6 %**); flag-raw production +0.007168 differs from all of these
  because the flag also swaps the grid (the §4-item-12 confound,
  correctly separated by the grid-only cell).
- **Where the 6 % VALUE gap becomes load-bearing (the actual gate killer):**
  see §4 — on the D_gen axis the increment is sensitive to the flag
  Σ-ratio's LEVEL, not just its slope, and the probe's baseline-arm value
  error flips the predicted increment's sign.

## 4. The D_gen (B-cell) axis — assembly validated, gate value retired

The B‴−B″ diagnostics give the flag's shift of the actual generator
normalisation per event: −Δ_flag ln D_gen(h), identical for the 1D and 2D
channels at 1e-14 (Z5 atomic-ledger exact, re-confirmed). Reconciliation
(p1c):

- The z2/z3 table assembly (n̂_w + β_Ḡ(zres)) fed with the PRODUCTION Σ
  tables reproduces the measured per-event shift at every venue h to
  ≤1e-3 relative (e.g. 0.73: 0.0035086 vs measured 0.0035063), and the
  B-cell 0.73→0.86 increment: **assembled +0.44 vs measured +0.45** (the
  readout's "+0.45 @0.86"). **The D_gen assembly arithmetic is correct.**
- The same assembly fed with the PROBE Σ tables gives the z3 gate value
  **−6.47**. The sign flip is the flag Σ VALUE ratio: probe
  shrunk/mz(0.73) = 0.975 (negative ln, decaying with h) vs production
  on/off(0.73) = 1.026 (positive ln, growing with h). Because
  Δ_flag ln D_gen(h) ≈ P_cat(h)·Δ_flag ln Σ(h) and P_cat falls from 0.139
  to 0.104 over 0.73→0.86 (measured, p1c), the increment is dominated by
  the product of the P_cat decline with the Δ_flag ln Σ LEVEL — the 6 %
  baseline-arm value error (§3) is amplified into the full −6.5-vs-+0.45
  gate discrepancy. **The −6.5 ± 4 gate value is retired: it was computed
  from a probe baseline arm whose value does not reproduce the production
  flag-OFF object, on an axis that amplifies exactly that error.**
- D_gen-axis clamp twin (probe machinery, p1b): increments vs the
  mz-binned baseline — full switch **−6.47**, unclamped-only **−1.20**,
  clamped-only **−5.26** ln. Further evidence of the axis's value
  fragility: merely moving the SAME estimator pair from the binned
  profile to row-by-row shifts the baseline gap by −1.46 ln and the full
  increment to −5.87 ln — 0.6-ln-scale swings from a 1 % value change.

## 5. Items (ii)/(iii) — convention deltas and shrinkage weights (bounds)

- **(ii) d_L convention (step vs linear on the same stored tables, p1b):**
  value ratio linear/step at 0.73 = 1.00084 (M_z-only) / 1.00062
  (shrunk-joint); dln(0.73→0.86) deltas +1.3e-4 / −3.0e-4 (D_gen-axis
  effect ≤ 0.2 ln). NEGLIGIBLE against the 4.9 % grid/interpolant item
  it is part of.
  The lifted-knot erf-sum convention (§3.3-C choice (a)) affects only the
  per-host 2D inner-M integrals, which are load-bearing in NEITHER A/B
  cell (diagnostic-only in `absolute_marginal` and `generator_marginal`);
  its bound is the doc's own (ln10·Δm)²/8 ≈ 1.5e-3 relative and it cannot
  contribute to the observed deltas.
- **(iii) per-cell shrinkage vs uniform w̄:** already measured by z3 on
  the gate axis: per-cell −6.47 vs uniform-w̄ −6.96 (7 % of the gate
  value, and the gate itself is retired per §4). Catalogue-weighted w̄ on
  the ACTUAL row-by-row query set = **0.8310** (p1a) vs the binned-profile
  0.8341 (z3) — 0.4 % apart. Not a parity breaker.

## 6. Overall verdict and the pre-registrable campaign prediction

**Verdict on the mandated question.** (d1) is NOT "suppressed by (g1)".
The measured production movement is genuine, matches the probe at the Σ
slope level to 6 %, and is ~75–90 % CARRIED by the clamped weight through
the u-dependence of the m_max boundary node. The ×3–5 shortfall against
the −6.5 gate decomposes into (a) a ×3.2 axis-translation error (D_gen
multiplier 718 vs A-cell multiplier ≈ 225, both measured) and (b) the
D_gen gate value itself being an artifact of the probe baseline arm's 6 %
value error (§4). Removing the m_max clamp does NOT predict (d1)
reappearing at ≈ −6.5 ln; the current (d1) effect on the A-cell axis is
≈ −1 to −2.2 ln and there is no measured mechanism by which the clamp-free
pool inflates it by ×3 or more — though the clamp-free pool re-measures
the dominant (clamped) 84 % of the weight with real data, so the movement
is genuinely unpinned in both directions beyond the band below.

**Pre-registrable prediction for campaign #51 (replaces the runbook
thread-1 sentence).** Stated at table level plus the measured translation,
on the campaign's OWN venue:

1. **Table level (primary, machinery-matched):** on the new pool
   (M ∈ [10⁴, 10⁷] source frame, ESS-floor sizing, w̄ → 1), the
   conditioning-only movement of the with-BH catalogue selection sum,
   Δ_cond dln Σ_glob_wbh(0.73→0.86) ≔ dlnΣ(joint) − dlnΣ(M_z-only, matched
   grid/interpolant), is predicted **positive** (joint conditioning
   steepens the h-slope — it held in every arm and machinery measured
   here) with central value **+0.0085, band [+0.003, +0.025]** (current
   measurement ±6 % parity, widened ×~3 for the support change: 84 % of
   the query weight moves from boundary-extrapolated to measured).
   Falsified if Δ_cond dln Σ ≤ 0 or > +0.03.
2. **Posterior level (A-cell / absolute_marginal):** Δ(2D ln @0.86,
   conditioning-only) = −[Σ_i s_i] · Δ_cond dln Σ with Σ_i s_i MEASURED on
   the new venue's diagnostics (this venue: ≈ 220–230; the new pool's
   deeper mass support may change it). With both factors at current scale:
   **≈ −1 to −2.5 ln — a genuinely small (d1)**. Falsified (and (d1)
   re-opened as a major owner) only if the measured conditioning-only
   A-cell increment < −5 ln.
3. **Mandatory controls carried over:** the grid-only cell (the confound
   was measured at the same order as the signal, opposite sign) and the
   per-event translation check of §1 (Σ tables → exact per-event
   re-prediction must close on the diagnostics as it does here).
4. **Retired:** the −6.5 ± 4 D_gen-axis gate and any A-cell band derived
   from it. Any future D_gen-axis gate must be computed with
   production-parity Σ VALUES (grid-matched baseline arm), because that
   axis amplifies value-level convention deltas (§4).

**Consequences for the residual attribution:** the ≈ +23 ln 2D HIGH
residual keeps (d2) + (g1)-as-support-limitation as owners; (d1) is
measured small on the production axis AND correctly predicted small — the
estimator, the implementation, and the probe all agree once the axis is
right. The campaign remains the 2D critical path for (d2)/(g1) reasons,
not in the hope of a hidden −6.5 (d1).

## Provenance of every number

- Cluster Σ_off: `cellApp/logs/evaluate_*.err` (jobs 6061150). Σ_on /
  Σ_grid-only: back-solved from `zmzApp` / `zmzGridOnly` vs `cellApp`
  `event_likelihoods.csv` per-event ratios (std < 1e-13). Local rebuild:
  `p1a_results.json` / `p1a_gridonly.json` (stack: working tree @
  a42cce4, production modules unmodified).
- B-cell shifts: `cellBpp` vs `zmzBpp` diagnostics; assembly:
  `p1c_results.json`.
- Probe rebuild + hybrids + conventions: `p1b_results.json` (validated
  against `z2_results.json` / `z3_results.json` at <1e-3).
- A-cell translations: exact per-event formula on `cellApp`
  `event_likelihoods.csv` (§1 table reproduces the ZMZ readout).
