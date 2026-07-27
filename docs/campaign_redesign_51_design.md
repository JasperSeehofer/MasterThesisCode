# Campaign redesign design doc (#51) — mass bounds + ESS-floor injection pool

**Status: FINAL (2026-07-28) — executed under the author's session directive
of 2026-07-28 ("authority to go autonomous … and even launch campaigns; one
clear place in the code where the boundary is set, scientifically supported;
no additional clamping"), which together with the ratified Amendments 1/2
constitutes approval. The bounds change landed as `[PHYSICS]` commit
`ecb56d6` (physics-change protocol applied; five gate items recorded in the
commit message and §1–§2 below). Items flagged [AUTHOR-REVIEW] are decisions
taken autonomously that merit explicit post-hoc review.**

Authority: FIX-3 ratification Amendments 1 & 2
(`docs/derivations/fix3_zmz_catalog_selection.md`, 2026-07-27), issue #51
(author-ratified directives), `RUNBOOK_NEXT_SESSION_4.md` thread 1.

Inputs feeding this doc:
- Recon report (this session, 2026-07-28): frame-convention pinning, bounds
  history, cost frame. Key citations inlined below.
- `results/lcat_h_dependence_20260725/mass_ab_20260727/P1_PARITY_AUDIT.md`
  — the mandated pre-campaign parity audit: (iv) REFUTED, gate retired
  (§6 below).
- `results/lcat_h_dependence_20260725/campaign_sizing_20260728/SIZING_ANALYSIS.md`
  — ESS-floor sizing analysis (§3–§5, §7 below).

---

## 1. The mass-bounds decision (Amendment 2)

### 1.1 What is wrong today (MEASURED, recon 2026-07-28)

1. **Draw-side override.** `cosmological_model.py:179–180`
   (`Model1CrossCheck._apply_model_assumptions`) unconditionally overwrites
   the `ParameterSpace.M` default `[1e4, 1e7]` (Babak et al. 2017 valid
   band, `datamodels/parameter_space.py:57–58`) down to `[10^4.5, 10^6]`.
   History: commit `cbe1a6f3` (2024-06-20, message "changed boundaries of
   mass") — a one-line change with no recorded justification.
2. **Double-duty constant → detector-frame wall at exactly 6.000.**
   `main.py:injection_campaign` lifts the source-frame draw to
   M_z = M·(1+z) and then **skips any injection with
   M_z > parameter_space.M.upper_limit** — the *source-frame* bound reused
   as a *detector-frame* truncation. Since M_z ≥ M, this cut dominates and
   produces the hard log₁₀ M_z = 6.000 wall in every z-tercile. The pool's
   cap frame convention is hereby PINNED: **detector-frame**, an artifact
   of constant reuse, not a deliberate bound.
3. **Uncoordinated handler constant.** `handler.py:27–28`
   (`M_min=1e4, M_max=1e6`) is used only by dead code
   (`draw_z_and_mass_from_gaussian`) and as an unused default; production
   pruning receives the cosmological model's bounds via `main.py:105–109`.
   Cleanup, not a physics fork.
4. **Consequence (numbers of record).** 81.4 % of the catalogue's
   R_eff-weighted rate-weight sits at log₁₀ M_z ≥ 6.0 (45.8 % ≥ 6.5; mean
   6.43) — the root of the p_det grid's (g1) clamp fraction. On the
   valid-band profile the weighted median is 6.56, p99 = 7.00,
   self-terminating by 2.5×10⁷ (issue #51 body).

### 1.2 The decision

- **Source-frame draw band: M ∈ [10⁴, 10⁷]** (delete the override; the
  `ParameterSpace` default becomes operative). Rationale: "as large as
  scientifically correct" (Amendment 2); the Babak M1 mass function is
  the model's own validity band and the catalogue's rate-weight
  self-terminates by 2.5×10⁷ — nothing physical justifies 10⁶.
- **Detector-frame truncation in `injection_campaign`: REMOVE** (or
  replace with the FEW waveform-validity guard if one exists at the
  sampler level — to be settled at implementation under
  `/physics-change`). The pool must cover the catalogue's detector-frame
  support m = log₁₀ M_z ∈ [4, ~7.4] (z ≤ 1.5).
- **Handler constants:** delete `handler.py:27–28` together with the dead
  `draw_z_and_mass_from_gaussian` (already slated — GitHub #7 notes
  `datamodels/galaxy.py` deletion; the handler dead path rides the same
  cleanup). Pruning continues to take bounds from the cosmological model.
- **Narrowing only if VERIFIED:** the upper bound may be pulled back only
  by the detectability pilot (§5) showing no detections above a candidate
  boundary.

## 2. Frame-convention specification (new, explicit)

To prevent recurrence of the double-duty bug, the campaign fixes ONE
convention, stated everywhere it matters:

- `ParameterSpace.M` limits are **source-frame**.
- `injection_campaign` writes the CSV `M` column in **detector-frame**
  M_z = M·(1+z) (unchanged, documented at `sdp.py:414–420`).
- Any truncation applied to M_z must reference a **detector-frame**
  constant with its own name and justification — never the source-frame
  draw bound. Default: no detector-frame truncation.

## 3. Grid re-noding on the new support

Per `SIZING_ANALYSIS.md` §6 (all numbers MEASURED on synthetic draws with
the probe/production kernel conventions):
- **61 u-nodes on [0, ln 2.5] × 69 m-nodes** on m = log₁₀ M_z ∈
  [4, 7.398] — 0.05 dex spacing, i.e. probe-parity *density*, not count.
- Bandwidths unchanged: Scott d=2 N^(−1/6) both axes, Abramson on u; ESS
  and w̄ measured insensitive to noding (31-m grid identical to 4
  decimals); the binding constraint is interpolation fidelity
  (spacing ≲ σ_m ≈ 0.08 at N = 200k). Storage ≈ 101 MB float64.

## 4. ESS floor, sampling measure, and total N (Amendment 1)

From `SIZING_ANALYSIS.md` (measured measure × N frontier, seed 20260728):

- **Sampling measure: stratified 3-component mixture `mix3_50_25_25`** —
  0.50 stratum 'a' (Babak M1 rate density on the widened box; keeps every
  pool-marginal leg exactly population-sampled at 2× the current pool),
  0.25 stratum 'b' (catalogue-coverage ∝ the R_eff-weighted `W_z_lm`
  profile; puts variance where the acceptance criterion scores), 0.25
  stratum 'c' (flat in (u, m) on the reachable region; principled,
  coordinate-derived guard lifting grid-wide min ESS ~10–20 → ~170). A
  recorded `stratum` column makes the split auditable; marginal legs use
  stratum 'a' only (measure-match rule Z1), the joint conditional
  S(d_L | u, m) is measure-free and uses all rows. Nothing is fitted to a
  desired answer; all component densities derive from repo objects.
- **Total N = 200,000 SNR-only injections.** Delivers reachable
  catalogue-weighted **w̄ = 0.9985** (shrinkage attenuation of the −15.6 ln
  joint increment ≈ 0.02 ln — inert), catalogue median ESS 8160, W-frac
  ESS<500 = 0.01 %, grid-wide reachable min ESS 172. Fallback if
  compute-constrained: 100k still passes (w̄ = 0.9974); 500k unnecessary.
  The status-quo pure-'a' measure can NEVER reach w̄ ≥ 0.99 at feasible N
  (0.947 at 10⁶; ESS grows ≈ N^0.59 because Scott bandwidth shrinks).
- **ESS floor = 1000** per catalogue-support node (⇒ per-node K5 weight
  ≥ 0.99; SE(Ŝ) ≤ 1.6 %).

Pre-registered acceptance criteria (Amendment 1, fixed here):
1. Catalogue-weighted median ESS ≥ 1000; catalogue weight-fraction on
   ESS < 500 nodes ≤ 1 %; reachable-weight w̄ ≥ 0.99 (shrinkage measured
   inert). Scored by rebuilding the joint grid from the delivered pool
   and publishing the §3.4-style table.
2. **[AUTHOR-REVIEW]** The w̄ criterion is defined on the *reachable*
   94.96 % of catalogue weight: 5.04 % sits above m = 7 + log₁₀(1+z)
   (source M > 10⁷ — outside the Babak validity band, structurally
   uncoverable by ANY measure). It is exempted by construction and must
   be reported separately, never hidden.
3. Joint-grid build logs (`.err` with the ESS diagnostic line) archived
   with the run (the 6065823–25 logs were not retained locally).

## 5. Detectability-verified-narrowing pilot

Measured status (`cap_analysis.json`, canonical 50k pool): the detection
horizon d_hor = SNR·d_L/20 *plateaus* into the current cap (p90 3.8→4.5→
4.5→4.1 Gpc over lm ∈ [5.5, 6.0]; top bin still detects 3.7 % with
d_hor max 6.2 Gpc ≫ the 1.82 Gpc catalogue depth). **Detections continue
above 10⁶ with near-certainty — narrowing is NOT verified and is not
assumed.**

Pilot (runs FIRST): **N = 2,000** SNR-only injections, log-uniform source
M ∈ [10^5.8, 10⁷] (half-bin overlap with the existing pool for
cross-validation), z from the measure-a z-marginal, detector-frame
ceiling 10^7.398 (the derived image — never 10⁷). Decision rule
(pre-registered): the upper bound may be narrowed to 10^lg* only if every
0.2-dex bin wholly above lg* has max d_hor < ½·d_L(z_min-cat, h = 0.60);
binomial backstop 0/1000 ⇒ P(det) ≤ 0.3 % (95 % CL). Expected outcome: no
narrowing; the pilot also documents any FEW waveform-validity limit above
10⁶ (a model limit, distinct from a detectability limit) and its rows are
recyclable as the first high-m campaign batch.

## 6. Pre-registered prediction carried into the campaign

The runbook-4 statement — *(d1) reappears at ≈ table size (−6.5 ln) on the
clamp-free pool* — is **RETIRED**, not carried. The mandated P1 parity
audit (`P1_PARITY_AUDIT.md`, 2026-07-28) refuted the (g1)-suppression
mechanism: clamped queries DO feel the u-conditioning (top m-node ESS
median 360) and carry ~75–90 % of the movement; the ×3–5 shortfall was an
axis-translation error in the z3 gate (D_gen multiplier 718 vs the
A-cell's measured Σᵢsᵢ ≈ 225) compounded by a 6 % probe-baseline value
error on an axis that amplifies value deltas. Probe, production, and
derivation agree once the axis is right (−2.35 predicted vs −2.20
measured).

Replacement prediction (audit §6, carried verbatim):
1. **Table level (primary):** Δ_cond dln Σ_glob_wbh(0.73→0.86) > 0 on the
   new pool, central **+0.0085**, band **[+0.003, +0.025]**; falsified if
   ≤ 0 or > +0.03.
2. **Posterior level (A-cell):** conditioning-only Δ(2D ln @0.86) =
   −[Σᵢsᵢ]·Δ_cond dlnΣ with Σᵢsᵢ measured on the new venue ⇒ ≈ **−1 to
   −2.5 ln**; (d1) re-opens as a major residual owner only if < −5 ln.
3. Mandatory controls: grid-only cell (same-order opposite-sign confound
   measured) + the per-event Σ-table translation check.
4. The ≈ +23 ln 2D HIGH residual attribution stays with **(d2) +
   (g1)-as-support-limitation** — the campaign is the critical path for
   those, not for a hidden −6.5 (d1).

## 7. Cost / walltime plan

Frame (recon 2026-07-28; per-task caps are receipts, totals partly OPEN):
- Simulate: `gpu_h100_short` (28 H100s, 30-min task cap), last campaign =
  500 GPU tasks (5 seeds × 100 tasks × 40 steps). Injection:
  `gpu_a100_short`, 30-min cap. Evaluate: `cpu,cpu_il`, 16 CPUs, 41-point
  h-grid arrays, measured 1.5–6 min/task incl. joint-grid build. Combine
  ~5 min.
- Per-user submit cap ≈ 300–550; orchestrator throttles at
  `MAX_PENDING=150` (`cluster/campaign_orchestrator.sh`).
- Workspace expires **2026-09-23** (last extension used). The campaign
  must complete retrieval well before that.

Scale plan: injection pool 200k = 4× the 50k pool ⇒ ~2000 A100 30-min
injection tasks (batched under the ~150-pending throttle; strata b/c
draws are cheaper than emcee but SNR evaluation dominates). Simulate +
evaluate stages unchanged in structure (5 seeds × 100 GPU tasks × 40
steps; 41-point evaluate arrays); simulate-stage SNR-pass rate on the
widened band to be measured by the pilot before final task-count lock.

## 8. Sequencing (as executed)

1. Amendments 1/2 ratified (2026-07-27); author autonomy directive
   (2026-07-28) — this doc records, rather than awaits, the decisions.
2. `[PHYSICS]` `ecb56d6`: single-source boundary + clamp removal (old
   behaviour pinned in `c12c295` per protocol; 1088 tests green).
3. Stratified-mixture injection sampling + stratum-aware estimator legs
   (opt-in `--injection_mixture`, default OFF = old behaviour).
4. Detectability pilot (§5) on the cluster — cheap, first cluster step.
5. Full injection → simulation → evaluation campaign per §4/§7, carrying
   the §6 replacement prediction and the §4 acceptance criteria.
6. Post-campaign: re-run the FIX-3 §7.1 A/B on the new universe (flag
   stays default OFF until the joint ship gate).

## Open items

- [AUTHOR-REVIEW] §4 item 2: w̄ acceptance defined on reachable 94.96 %
  of catalogue weight (5.04 % above the Babak band is structurally
  uncoverable) — decision taken autonomously, review requested.
- OPEN (recon): total node-hours of the last campaign not locally
  reconstructible (no simulate logs/`sacct` retained) — cost plan uses
  task-cap arithmetic instead.
- OPEN (recon): the 81.4 % figure could not be exactly re-derived from
  `catalog_zw_profile.json` alone (recompute brackets 51–88 % depending on
  overflow-bin treatment); 81.4 %/45.8 %/6.43 remain the numbers of
  record (derivation doc, 5× verbatim); the 87 % figure in #51 is the
  valid-band-profile variant. Not load-bearing for the decision (§1.2
  stands on Amendment 2 regardless).
