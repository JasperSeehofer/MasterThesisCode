# A8-v2 ADVERSARIAL VERIFIER ADDENDUM — PREREGISTRATION_PRODCAL_LADDER.md

**Verifier:** independent A8-v2 adversarial pre-registration verifier (precedent: ledger row #110
item 2). Read-only on all files except this one. Two passes: Part I against DRAFT v1
(2026-08-17), Part II the delta re-check against DRAFT v2 (amendments 1–8 applied). All numbers
recomputed from the cited on-disk record, never taken from the prereg's own text.

---

# PART I — v1 verdict (of record; drove amendments 1–8)

## Checklist verdicts (v1)

### 1. BANDS — DEFECT, MAJOR
- Reproduced: SE(cov68)=√(0.68·0.32/120)=0.0426; PASS cov band [0.594,0.766] exact; #67
  cov68=0.38/0.12 at n=1000/4000 for a +0.0022-class bias (`pp_coverage_noisemodel_20260711/
  SUMMARY.md:74`); H-P/N-2 band mutual exclusivity; Block A/B cell counts and budget arithmetic.
- Did NOT reproduce: **§2's σ(n=250)≈0.004 is not in the #67 record.** Measured from
  `pp_coverage_noisemodel_20260711/*.json` (`map_std`): 0.0051–0.0074 at n=250 (deep cells),
  0.0027–0.0042 at n=1000, 0.0017–0.0021 at n=4000 — **sub-1/√n** improvement (n=4000 sits ~25%
  above the 1/√n extrapolation from n=1000). Interpolated σ(1600)≈0.0025 ⇒ SE(bias)≈2.3e-4 at
  R=120, so the registered "~+0.0005 residual is a ≥3σ read" was actually **~2σ**; the "≥6σ" H-B
  figure was likewise unsupported.
- The v1 N-3 band [−0.0045,+0.0010] did not follow from its stated recipe (record gives
  −0.0030…−0.0010 ± 2·SEM≈0.0011 ⇒ [−0.0041,+0.0001]).
- H-B band ill-defined when the (never-measured) V-deep const floor is small or sign-mixed;
  needed a registered precondition. H-N "bias → 0" had no numeric threshold.

### 2. CONFOUNDS — DEFECT, MAJOR
- **N-1 was unexecutable inside the registered grid**: mass channel requires `catalogue_mode`
  and `selection_cell` requires the mass channel (`test_pp_coverage_mass.py:191,197`), while the
  07-11 baselines are pre-`catalogue_mode`, mass-free, continuum-mode runs (seed 20260701).
  With mass ON at both venues, no registered cell was comparable to the 07-11 record.
- **N-3's absolute control band does not transfer** from the mass-free continuum venue to a
  mass-ON catalogue venue; only the paired sign-flip read vs the `off` twin is robust.
- Builder's ambiguity 4 (S̄_φ replaces p_det whenever mass is on) reinforces both: even the
  selection-`off` cell changes its denominator object once mass is on.

### 3. DECISIVENESS — OK with one MINOR gap
H-P FAIL + N-3 PASS routes unambiguously to the FAIL branch; N-1/N-2 STOP rules take
precedence. Unassigned pattern: **H-P PASS + N-3 FAIL** (the #66 phenomenology recurring) —
needed a registered MIXED branch with a separating cell.

### 4. INSTRUMENT RISK — DEFECT, MINOR
Unit-level structural nulls are real and pinned (α_M=0 reductions, σ_cond→0 limit,
channel-locality, golden byte-identity, paired RNG stream, three-way noise-cell distinctness;
28 collected tests confirmed). All CLI knobs needed by the design exist (`--noise-model
{const,model,production}` at `pp_coverage.py:2752-2753`, `--selection-cell` off/1d/2d/fused,
`--mass-channel`/`--mass-horizon-index`, `--n-events`, `--n-galaxies`, `--sky-frac`).
The dangerous H-P false-PASS direction is a **silently-inert lever** (fused or mass channel not
engaged at campaign configuration); N-1/N-2 do not guard it and N-4 "no degradation" is exactly
what an inert lever produces. Cheapest guard: campaign-level engagement nulls on the paired
per-realization deltas (now N-5).

### 5. DISCIPLINE — DEFECT, MAJOR
- **v1's seed scheme was unimplementable and self-defeating**: the instrument takes one master
  `--seed` per invocation and spawns realization RNGs internally (`pp_coverage.py:2474,2499`);
  per-cell distinct seeds would have UNPAIRED every cross-cell comparison the power section
  depended on. Seeds must vary only over (venue × n), shared across the noise × selection axes.
- v1 base 20260818: no pp_coverage campaign ever used it (all 07-10/07-11 RUNBOOKs: 20260701),
  but the emitted range 20260818–20260937 collided with per-task seeds of the sibling coverage
  harnesses (`closed_loop_gfrac_20260805` 20260805–20261004, `calibration_gate_20260808`
  20260808–20261207, `calibration_gate_v2` R0 20260805–20261004).
- **`readout_prodcal.py` did not exist** at v1 despite being same-commit-REQUIRED.
- `h_step` was unregistered.

### 6. SCOPE — OK on production-change; Q-0 UNDETERMINED pending verification
"No production change is proposed by any branch" explicit; nothing re-litigates the exoneration
union (`CLAIM_2D_BIAS_20260730.md` §Exonerated + ledger §2); N-3 is a calibration read, not a
mechanism re-open, matching the ratified D-3 [RULE] (row #120 item 3). But **Q-0 (UNPAIRED) is a
same-session [AGENT] product used load-bearingly as the definition of the `production` noise
cell**; its verdict had been neither adversarially verified nor recorded — required an
independent file:line verification before commit.

### 7. VENUE PARAMETERS — CONDITIONAL-OK
Deferring V-deep parameters was acceptable only if §7 is filled in the same commit, before any
run, with quoted production-diagnostic anchors, and any later edit voids the registration.

## v1 REQUIRED AMENDMENTS (1–8, all blocking except 6–8)
1. §2 power restatement on the record's `map_std` numbers; residual-class read demoted to a
   ~2σ BOUND; n=1600 cov68 promoted to primary [R-2] discriminator.
2. Pairing-preserving seed scheme (one master seed per (venue,n), shared across
   noise × selection), base outside every consumed range.
3. H-B registered precondition (V-deep const+fused floor ≥ +0.0015, sign-coherent, > +2·SE per
   truth; else UNDETERMINED-BY-DESIGN, unscored).
4. (a) Dedicated mass-OFF continuum Block N1 replication cells at seed 20260701; (b) N-3 PASS
   re-based on the paired read vs the `off` twin; corrected 07-11 band demoted to
   reported-not-scored reference.
5. Scorer committed same-commit, computing exactly the §3 statistics + the N-5 engagement nulls.
6. H-P PASS + N-3 FAIL registered as first-class MIXED with the V-ctrl mass-off separating cell.
7. Independent Q-0 verification (file:line quotes) recorded in §7.
8. (a) numeric H-N PASS; (b) N-1 SE definition; (c) `--h-step` registration; (d) §7 header
   tightened to same-commit; (e) N-3 band arithmetic shown.

**v1 gate: GO-WITH-AMENDMENTS (1–5 blocking).**

---

# PART II — DELTA RE-CHECK (DRAFT v2 + `readout_prodcal.py`)

## II.1 Amendment-by-amendment verification

| # | status | notes |
|---|---|---|
| 1 | **APPLIED, one residual overclaim** | §2 now carries the record's numbers verbatim (0.0051–0.0074 / 0.0027–0.0042 / 0.0017–0.0021; σ(1600)≈0.0025; SE 2.3e-4), the ~2σ-BOUND framing, and cov68 as primary [R-2] discriminator — all correct. **Residual:** "a 2× reduction is a ≥3σ paired read at worst-case correlation" does not reproduce: at the PASS edge (floor exactly +0.0015 ⇒ delta threshold 0.00075) with worst-case paired SE √2·2.3e-4≈3.3e-4, the read is **2.3σ**, not ≥3σ. It is ≥3σ only for floors ≥ +0.002. Same defect class as v1 — must be fixed (D-6 below). Also: §3 attributes H-B to Block A (n=250), but §2's H-B power arithmetic is at n=1600; at n=250 the PASS-edge read is ~1σ. H-B must be scored on the Block B n=1600 const/production × off/fused cells (D-6). |
| 2 | **APPLIED and verified** | Single master seed per (venue,n), shared across noise × selection — matches the instrument interface and preserves pairing (pinned by `test_no_scatter_keeps_the_random_stream_aligned`). Freshness of base 20270818 independently confirmed: grep over `results/` finds **no** 202708xx value in any prior artifact (the 2027xxxx integers in `calibration_gate_20260808` are outside 20270800–20270899). Block N1's deliberate 20260701 reuse is correctly flagged. |
| 3 | **APPLIED** | Precondition in §1 and in the §4 H-B row, with UNDETERMINED-BY-DESIGN unscored. |
| 4 | **APPLIED** | Block N1 registered (mass OFF, continuum, 07-11 parameters, seed 20260701, 0.5 CPU-h, decides N-1); N-3 PASS is now the paired read; the v1 band is explicitly withdrawn and the corrected reference band [−0.0041,+0.0001] matches my recomputation. |
| 5 | **APPLIED with two gaps** | Scorer exists and is faithful (see II.2). Gaps: registered-pair manifest missing; N-5 direction check needs a numeric band (see II.2, D-3/D-4). |
| 6 | **APPLIED** | First-class MIXED with the V-ctrl mass-off production+fused separating cell. |
| 7 | **APPLIED and independently re-verified by this verifier** | All three quotes check out at current working tree: (i) `darksiren_emri/datamodels/detection.py:133-136` — `self.d_L = parameters["luminosity_distance"]`, `self.d_L_uncertainty = np.sqrt(parameters["delta_luminosity_distance_delta_luminosity_distance"])` (injected truth + CRB diagonal); (ii) `convert_to_best_guess_parameters` has its definition at `detection.py:161` and **zero call sites** in `darksiren_emri/`; (iii) `bayesian_statistics.py:3613` — `_log_norm_3d[slot] = -0.5*(3*log(2π) + logdet_3d)`, precomputed once, constant in z and h. Q-0's UNPAIRED reading survives a third independent read. §6 caveat 5 (residual third-fact risk) is honest. |
| 8 | **APPLIED, one gap** | (a) H-N numeric ✓; (b) N-1 same-seed plain-SE ✓; (c) h-step registered for Block A/N1 (0.004) and n=1600 (0.001) but **silent for the n=800 cells** (D-5); (d) §7 header tightened ✓ (but see II.3 on the carve-out wording); (e) band arithmetic shown ✓. |

## II.2 Scorer audit (`readout_prodcal.py`)

- **Statistic list matches §3/§4**: per cell × truth × channel — map_bias mean ± SE
  (map_std/√R), cov50/68/90 ± binomial SE (nominal-p convention, consistent with §2),
  rail_fraction, 2D block when present; per pair — [A2] per-realization delta mean ± SE,
  quartiles, n_pairs, and the **N-5 degeneracy flag** (`np.all(delta==0)`), with
  not-computable pairs surfaced as explicit `None`, never skipped. Verified against the
  harness's actual output keys (`pp_coverage.py:2557-2584`: `coverage`, `rail_fraction`,
  `map_bias`, `map_std`, `maps`, `mass_channel_2d.maps`) — compatible. Nothing extraneous is
  emitted that could smuggle an unregistered statistic into the verdict.
- **Gap 1 (D-3): the registered pair list is not pinned anywhere.** `--pair` is free at readout
  time, i.e. WHICH paired comparisons exist is currently a post-data choice. The §4 bands name
  the verdict-bearing pairs implicitly, but S-2 promises "every cross-cell comparison" and N-5
  quantifies over "all registered paired deltas" — quantifiers with no registered enumeration.
  Fix: a `PAIRS` manifest (cell_id tuples) either as a constant at the top of the scorer or as
  a table in §7; the scorer invocation of record uses exactly that manifest.
- **Gap 2 (D-4): the V-ctrl #66-direction positive-shift check.** Answer to the coordinator's
  direct question: it does **not** need to live inside the scorer as a boolean — the scorer
  emits `delta_mean` and `delta_se` per registered pair, so the directional read is a scored
  comparison **of scorer outputs**, which satisfies "no statistic outside the scorer enters the
  verdict." What it DOES need is a numeric band in §4, which is currently missing ("show the
  #66-direction positive shift" is unquantified): register PASS as channel-1d
  delta_mean(const+fused − const+off, V-ctrl) > 0 with delta_mean ≥ 2·delta_se at every truth
  (the #66 record's +0.006-class shift makes this an easy bar if the lever is live), FAIL
  (instrument suspect, STOP) if degenerate or if the shift is ≤ 0 at every truth, MIXED else.
  Adding a convenience boolean to the scorer is recommended (one line) but not required.
- The H-B precondition and all §4 bands are computable mechanically from scorer output ✓.

## II.3 New v2 content — new attack surface (three findings, two of them defects)

- **D-1 (MAJOR — unexecutable registered cell): V-ctrl as registered violates the instrument's
  own constraints.** §7 registers V-ctrl as "z_support = None (untruncated), mass ON" — but the
  mass channel requires `catalogue_mode`, and `--catalogue-mode` **requires `--z-support` and
  `--mixture-mode ∈ {lcat, absolute, generator_marginal}`** (`pp_coverage.py:2656-2664`;
  enforced, per the v1 finding on `test_pp_coverage_mass.py:191,197`). An untruncated mass-ON
  catalogue cell cannot be launched as registered. Fix: register V-ctrl with **z_support = 1.5**
  (= the harness's Z grid ceiling, making the truncation non-binding — the operative meaning of
  "untruncated"), and register V-ctrl's `n_galaxies` and `sky_frac` explicitly (currently
  absent; without them the control venue is underdetermined).
- **D-2 (MAJOR — estimator-defining knob unregistered / template unexecutable): the §7 CLI
  template omits `--mixture-mode` and `--z-support`.** As written the command errors
  (catalogue-mode's requirements unmet; default mixture is `two_branch`). Worse than the crash:
  `--mixture-mode` selects the ESTIMATOR (`absolute` is the production `absolute_marginal`
  analog per the intake §4) — leaving it to the "verified at execution, mismatch = deviation"
  clause converts an estimator choice into a post-registration degree of freedom. Fix: add
  `--mixture-mode absolute --z-support {zs}` (and the venue's sky_frac/n_galaxies) to the
  template, plus the Block N1 command template (continuum, no catalogue flags, per the 07-11
  RUNBOOK lines).
- **Pretuning procedure — forking-paths audit (D-7, blocking in two sub-parts).** The
  construction (targets registered before any run; archived, never scored; sole-permitted
  post-commit fill-in) is the right shape, and the anchor targets independently REPRODUCE: I
  recomputed from the named CSVs
  (`run_20260817_fusion_counterfactual/{fused_iiib,fused_joint_r1}/simulations/diagnostics/
  event_likelihoods.csv`): n_events = 1588 ✓, catalogue-bearing fraction at mid-grid h=0.725 =
  **0.618 / 0.690** ✓ (exact), mean g_frac = **0.371** in both venues ✓ (exact). Remaining
  exposure: (i) **the pretuning seed is unregistered** — if pretuning reuses a campaign seed,
  the venue is chosen after seeing (R=8-noisy) MAP/coverage output on realizations later
  scored; register a disjoint fixed pretuning seed (e.g. 20270999). (ii) **"first pair to
  land" is only mechanical if the candidate sweep is fixed** — register the candidate list and
  order (e.g. z_support ∈ {0.25, 0.30, 0.35} × sky_frac ∈ {1e-4, 2e-4, 4e-4}, lexicographic)
  or the fill-in is a choice, not a procedure. (iii) MINOR wording: the header's absolute "any
  later edit to §7 voids the registration" contradicts the registered carve-out — add "except
  the single registered pretuning fill-in (§7), itself append-only" to the header. (iv) Noted,
  acceptable as a disclosed analog: the pretuning target (harness `host_in_ball_fraction`) and
  the production anchor (fraction of events with L_cat_no_bh > 0) are analog estimands, not
  the same quantity — one clause in §7 saying so keeps it honest.
- **D-5 (MINOR but budget-coupled): n=800 h-step unregistered, and the Block B budget was not
  re-measured after amendment 8c.** The h grid is h ∈ [0.600, 0.860] (`pp_coverage.py:632-633`):
  66 points at 0.004 vs 261 at 0.001 — the per-h estimator recompute (the g_sel scan) is the
  dominant cost, so the ~8.5 s/realization anchor (measured at the default step) scales to
  roughly ~4× at the registered 0.001, putting Block B's n=1600 half alone at ~12–14 CPU-h and
  the campaign plausibly **over the binding 15 CPU-h ceiling** — which converts my own
  amendment into a mid-campaign STOP. Fix before commit (mechanical): register n=800 at 0.004;
  re-measure ONE n=1600/h-step-0.001 realization and refill the Block B line — or register
  0.002 for n=1600 (2× cost, comfortably in budget), citing the #67 fine-grid confirm (bias
  identical to ±0.0001 between 0.004 and 0.001) as the fidelity warrant.

## II.4 Consolidated DELTA amendments (D-1..D-7; all mechanical, exact text above)

1. **D-1 (BLOCKING)** V-ctrl: z_support = 1.5 (non-binding truncation) + register V-ctrl
   n_galaxies and sky_frac.
2. **D-2 (BLOCKING)** CLI template: add `--mixture-mode absolute --z-support {zs}` + venue
   flags; add the Block N1 command template.
3. **D-3 (BLOCKING)** Registered PAIRS manifest (scorer constant or §7 table); scorer
   invocation of record uses it.
4. **D-4** §4 N-5 row: numeric band for the #66-direction check (delta_mean > 0 with
   ≥ 2·delta_se at every truth, 1D channel, V-ctrl const pair); scorer boolean optional.
5. **D-5 (BLOCKING)** Register n=800 h-step (0.004); re-measure and refill the Block B budget
   at the registered steps, or register 0.002 for n=1600 with the #67 fine-grid warrant.
6. **D-6** §2: correct "≥3σ at worst-case correlation" to "≥2σ at the precondition edge
   (+0.0015 floor), ≥3σ for floors ≥ +0.002"; §3: assign H-B's scored read to the Block B
   n=1600 cells (Block A's n=250 H-B read is descriptive only).
7. **D-7 (BLOCKING)** Pretuning: fixed disjoint pretuning seed (e.g. 20270999) + registered
   candidate list and sweep order; header carve-out wording; one-clause analog-estimand
   disclosure.

## II.5 Independent verifications performed this pass (for the record)

- V-deep anchors recomputed from the counterfactual CSVs: **exact match** (0.618 / 0.690 /
  0.371 / 1588).
- Q-0's three §7 quotes re-verified at file:line by this verifier (distinct from both the Q-0
  auditor and the harness builder — Q-0 now carries three independent reads).
- Seed base 20270818: no collision anywhere in `results/`.
- Scorer smoke-audited against the harness's actual JSON schema (keys verified at
  `pp_coverage.py:2557-2584`).
- 28 collected tests confirmed (`pytest --collect-only`); all design-required CLI knobs
  confirmed present; `if __name__ == "__main__"` entry point confirmed (`pp_coverage.py:2860`).

---

## FINAL GATE

**GO-WITH-AMENDMENTS — D-1, D-2, D-3, D-5, D-7 blocking; D-4 and D-6 required but non-blocking
wording/band fixes. Do not commit or launch any cell until all seven are applied.** Every delta
amendment is mechanical with exact replacement text given above, and none touches a hypothesis,
a band's decision logic, or the branch structure — so **no further verifier pass is required**
if they are applied verbatim; the author's [DO] may be taken on the amended file directly. If
any amendment is applied other than verbatim (in particular D-1's venue values or D-7's
candidate sweep), that specific delta returns for a one-item re-check.

*Verifier addendum complete. This file is the only file written by the verifier; the prereg,
scorer, instrument, and all evidence files were read-only in both passes.*

---

# PART III — SCOPED ONE-ITEM RE-CHECK (DRAFT v3, 2026-08-17)

Invoked under Part II's verbatim-deviation clause for the three deltas applied other than by
exact replacement text: D-1's venue values, D-5's budget/ceiling, D-3/D-4's manifest and band.

## Item 1 — V-ctrl venue values (D-1): **OK**

n_galaxies = 200000, sky_frac = 1e-4, z_support = 1.5 (non-binding), matching V-deep's
candidate density so the control differs from V-deep on the depth/truncation axis only, is
accepted. Rationale for acceptance: (a) the single-axis contrast is the cleanest control
design available — the 07-11 controls were continuum-mode, so no exact catalogue-mode target
exists and "match V-deep except depth" is the least-arbitrary registration; (b) the choice is
non-critical by construction, because every V-ctrl verdict-bearing read (N-3, the D-4
#66-direction check) is PAIRED against its own same-venue twin — absolute venue idiosyncrasy
cancels, which is exactly why Part I re-based N-3 as paired. Registered before any run, in the
freezing commit: discipline satisfied.

## Item 2 — budget re-measure + ceiling 15 → 18 CPU-h: **OK**

Arithmetic verified: n=1600 half = 4 configs × 120 R × 3 truths × 23.5 s = 33,840 s ≈ 9.4
CPU-h ✓; n=800 half ≈ 1.7 CPU-h ✓; total ≈ 14.6 CPU-h ✓. The 23.5 s figure is a MEASURED
h-step-0.002 fused realization (vs my ~2× scaling guess of ~17 s — measurement beats
extrapolation, and using the fused/most-expensive config for all four configs is
conservative). The ceiling raise is acceptable: the ceiling binds from commit, this is a
disclosed PRE-commit re-registration grounded in a measurement (row #116 fill-at-submission
discipline), and it removes the exact mid-campaign-STOP failure mode D-5 flagged; 18 CPU-h
leaves ~23% margin over the measured total. Not a scope creep: still single machine, no
cluster.

## Item 3 — PAIRS manifest + `--registered` mode (D-3) and the N-5 band (D-4): **OK**

- Manifest enumerated and counted: 18 pairs ✓. Coverage against every §4 quantifier verified:
  N-5's (fused−off) and (const−production) exist for every venue × n (V-deep 250/800/1600,
  V-ctrl 250, both noise models, plus the model fused−off twins at both venues); the H-B
  scored pair (vdeep_1600_const_fused, vdeep_1600_production_fused) present, with its `off`
  and n=250/800 descriptive twins; the N-3 pair (vctrl_250_production_fused/off) present; the
  D-4 pair (vctrl_250_const_fused/off) present. No §4 band references a pair outside the
  manifest.
- `--registered <cells_dir>` implemented as specified: scores every cell file present and
  exactly the manifest, with missing pairs surfaced in `registered_pairs_missing` (never
  silently skipped) ✓; `--pair` demoted to exploratory/never-verdict-bearing in both the
  prereg and the scorer help text ✓. One non-blocking observation for the execution log:
  in `--registered` mode any additional exploratory `--pair` output lands in the same
  `pairs` array as manifest pairs — the invocation of record should therefore carry NO
  `--pair` flags (recommend noting this in the VERDICT section's invocation line).
- D-4 band in §4 row N-5 matches the Part II prescription exactly: PASS = channel-1d
  delta_mean(const+fused − const+off, V-ctrl) > 0 with ≥ 2·delta_se at every truth; FAIL =
  any degenerate delta OR shift ≤ 0 at every truth (STOP); MIXED = positive but < 2·delta_se
  at some truth ✓. Computable purely from scorer output ✓.

Also confirmed in passing on v3: the header carve-out wording (D-7 iii), the §2 H-B power
correction and Block-B scored-read reassignment (D-6), the executable CLI templates with
`--mixture-mode absolute` registered as estimator-defining (D-2), and the pretuning
discipline block (fixed disjoint seed 20270999, lexicographic 3×3 candidate sweep,
analog-estimand disclosure) (D-7 i/ii/iv) — all as prescribed.

## FINAL GATE (one-item re-check, 2026-08-17)

**GO.** All Part I and Part II amendments are now applied and verified (verbatim or accepted
per this re-check). No open verifier findings remain. The prereg + scorer are ready for the
author [DO]; on approval, commit `PREREGISTRATION_PRODCAL_LADDER.md`, `readout_prodcal.py`,
and the frozen harness in one commit, run the pretuning fill-in, then execute per §3. The
append-only line binds from that commit; any deviation is recorded in the VERDICT section.

---

# PART IV — PRE-CHECK OF PROPOSED AMENDMENT-1 (pretuning sweep extension; prereg committed at fe72d52b; 2026-08-17)

Situation of record: the registered 3×3 candidate sweep exhausted without landing —
host_in_ball_fraction 0.279/0.415/0.534 at z_support 0.25/0.30/0.35, monotone undershoot of
the [0.60, 0.70] target; sky_frac verified live on its designed levers (mean_ball_size
0.91→7.17, impostor_fraction 0.693→0.925) and inert on host-in-ball (a pure z_support
truncation effect). **Zero scored cells have run.** The registered D-7 procedure had no
exhaustion branch — a gap in the Part II prescription itself, now exposed; the extension
repairs a procedure defect, it does not react to any outcome.

## Question 1 — appended-amendment form vs append-only discipline: **OK; no v4 file**

A dated AMENDMENT-1 section appended below the VERDICT line respects the letter of the
committed rules (no edit above the VERDICT line; the §7 body untouched — the carve-out fill-in
blanks stay blank until the extended sweep lands). It also respects the substance, on three
grounds that must all hold and currently do: (a) no scored cell has run, so no outcome exists
to react to; (b) the only data consulted are the pretuning tuning-target fields, which the
registered procedure itself reads, from cells run at the registered disjoint seed 20270999;
(c) the extension preserves every discipline element (same seed, same targets, same
first-to-land rule, deterministic appended order). A superseding v4 file is NOT required and
would be worse: it would orphan the committed hash fe72d52b that the freeze chain cites. Three
form conditions, all cheap: (i) commit AMENDMENT-1 BEFORE any extension pretuning cell runs;
(ii) the amendment text states explicitly that no MAP/coverage field of any archived pretuning
output was read — only the tuning-target fields; (iii) the VERDICT section remains
readout-only and its eventual text cites AMENDMENT-1.

## Question 2 — the extension itself: **OK with two required additions; do NOT pin sky_frac**

- **Keep the sky_frac sweep.** Pinning it to 1e-4 would be wrong: landing requires BOTH
  registered targets, and while host-in-ball is sky_frac-inert, the completion-fraction target
  [0.30, 0.42] has not been shown to be. If (0.40, 1e-4) misses on completion share, the rule
  must be able to advance deterministically to (0.40, 2e-4) etc. If both targets turn out
  sky_frac-inert, the sweep is harmless (the order is fixed). Lexicographically taking 1e-4
  when it lands is not a defect — it is the registered rule doing its job.
- **Required addition 1 — no new tuning targets, but disclose the loading.** Impostor loading
  (ball ~2.5–3, impostor_fraction ~0.75 at zs=0.40/sf=1e-4) is NOT a registered target, and
  adding one now — after seeing the nine pretuning outputs — would be exactly the post-hoc
  tuning-rule expansion the discipline forbids. Instead the fill-in line must record the
  frozen pair's mean_ball_size and impostor_fraction as DISCLOSED descriptive facts, carried
  under the §6 venue-transfer caveat (the harness venue's impostor loading vs production's is
  then an on-record transfer axis, not a hidden one).
- **Required addition 2 — exhaustion clause.** The slope (~+0.12 in host-in-ball per +0.05 in
  z_support) projects ≈0.65 at zs=0.40 — likely to land — but the amendment must close the gap
  it is repairing: if the extended sweep ALSO exhausts, execution stops and returns to the
  author; any further extension is an AMENDMENT-2 under this same pre-check discipline (no
  silent iteration).
- Cost: six R=8/n=250 pretuning cells are budget-negligible; no change to the 18 CPU-h
  ceiling needed. Seed 20270999 reuse across candidates is the registered design ✓.

## GATE (AMENDMENT-1 pre-check)

**GO — conditional on the amendment text incorporating the two required additions (loading
disclosure in the fill-in line; exhaustion clause) and the three form conditions of Question 1
verbatim.** So amended, AMENDMENT-1 is a procedure repair executed before any outcome existed,
fully inside append-only discipline; no superseding prereg is required, and no further
verifier pass is needed unless the extended sweep also exhausts.
