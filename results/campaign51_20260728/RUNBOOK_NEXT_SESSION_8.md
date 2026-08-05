# Runbook — next session (written 2026-08-05, end of the overnight autonomous session)

Supersedes `RUNBOOK_NEXT_SESSION_7.md` (its task queue is DONE: the research cycle is
established and amended A1–A4; gate (vii) was read properly; D1 ran as the first full
cycle to a bounded-null verdict; the §2 calibration gate was built and run). Session
ran dev box + cluster under the author's overnight mandate (2026-08-04/05): physics
work measured via default-off counterfactuals and pre-registered runs; **no production
formula was changed** — everything below that says "author" is the morning queue.

**Read in this order:**
1. This file §0–§2.
2. `.planning/derivation-gfrac-20260805/GFRAC_DERIVATION_PACKAGE.md` (Gate-B-amended)
   + `GATEB_REFUTATION_REPORT.md` — the g(h) verdict and its adjudication.
3. The four pre-registrations with appended verdicts:
   `results/run_20260804_postfix/gate_vii/PREREGISTRATION_FROZEN_GFRAC.md` (CONFIRM),
   `results/closed_loop_gfrac_20260805/PREREGISTRATION.md` (MIXED/not-REFUTE),
   `results/run_20260804_postfix/gate_vii/PREREGISTRATION_N2_SEL1D.md` (MIXED-bounded),
   `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_D1_SAND_REWEIGHT.md`
   (MIXED/bounded-null + the N2 discovered fact).
4. `BIAS_HISTORY_LEDGER.md` §1 rows 90–94 (August thread) + §4 new open threads.
5. Book ch12 (`book/site/ch12-bias-resolution.html`) — the living narrative.

---

## 0. State of the physics (2026-08-05, post-overnight)

The night's five verdicts compose into one picture:

| # | measurement | verdict | consequence |
|---|---|---|---|
| 1 | gate (vii) proper read | catalogue-leg tilt GREW −504.8→−604.8 (A2 exoneration void); composition-dominated (81% from 316 scatter-resurrected events) | catalogue leg is a muted DOWN-pull, not the 2D driver |
| 2 | frozen-g counterfactual (pre-reg CONFIRM) | g_frac(h) carries the whole residual 2D displacement (0.780/0.800 → 0.660/0.640; live == proxy) | the carrier is identified |
| 3 | g(h) derivation + Gate B | **CORRECT PHYSICS survives adversarial review** — the spectral-siren term, analytic closure 0.1–0.2%; population check ADVERSE to defect (implemented tilt 8% too small) | the carrier is (candidate-)legitimate; §7 re-attribution REFUTED (never-add-MAPs) |
| 4 | closed-loop 2-channel calibration, 200 seeds | MIXED/**not-REFUTE**: Δ2 = +0.011±0.004 (posterior-mean +0.005) | the production +0.05–0.07 2D displacement is NOT reproduced when the universe follows the estimator's assumptions |
| 5 | N-2 sel-1d counterfactual (pre-reg) | MIXED-bounded: chord +24.6/+22.7 in band; **1D stays railed 0.600 both venues**; sign-coherence 0.71 < 0.90 | N-2 is a real, positive, BOUNDED correction — NOT the rail owner |
| 6 | D1 three-arm S_and re-weight (pre-reg) | MIXED/bounded-null: m_S=0.032, m_R=0.011 (≪0.25); **g_frac bit-identical under S_and** | D1 does not reach the core object via the tilt route; the D1→g_frac (C7) convergence route is DEAD |
| 7 | D1 B2 | s_G/s_D NOT h-flat (Δln −0.0342, 6.8× band) | D1's tilt route was live and is now measured small (row 6) |

**Discovered facts (new, un-adjudicated — next intake queue):**
- **f_k is pool-fed into the catalogue leg**: under `volume_deconv`/`absolute_marginal`
  the catalogue-leg host-z prior carries the completeness callable f_k built from the
  injection pool — the D1 prereg's "L_cat carries no selection" assumption is FALSE.
  (Found via the N2 null failing loudly; 62–69% of L_cat_no_bh cells move under a pool
  substitution.) Un-modeled selection coupling into L_cat = candidate new mechanism class.
- **S1 joint-venue Σ⁴ᴰ self-cert failure (12.5%)** under A2 (iiib passes at 6.8e-5);
  β_Ḡ^φ passes both. Likely legitimate venue dependence via f_k — but unadjudicated.
- **kappa_cap kink at M=1e5 is ACTIVE** (event 953 straddles it; s_dex flips −0.43→+0.07
  below). Any g-related pin must be kink-aware (P1 was repaired accordingly).
- **N-1**: gate (i) is near-vacuous as measurement (2D catalogue leg identically zero for
  81.5%/61.8% of events); the algebraic completion-leg invariance proof supersedes it.
- Closed-loop 1D rails 200/200 (info-starved venue) and the numerator-selection
  diagnostic removes it 0/50 — sharp N-2-structure signature IN THE LOOP, but the
  production counterfactual (row 5) shows the production rail has a bigger owner
  (photo-z per the standing root-cause finding remains the leading account).

**Where the bias stands:** the 2D channel's displacement is (per rows 2–4) carried by a
physically correct spectral-siren term acting on top of a 1D channel whose rail is the
real anomaly; the rail's owner is STILL AT LARGE — N-2 and D1 are both real-but-bounded;
photo-z information starvation (h0-railing root cause of record) remains the standing
account, now with the calibration-gate instrument available to test it.

## 1. Morning author queue (decisions, in proposed order)

1. **R-A (g(h) ruling):** accept "correct physics" for g_frac's h-slope (Gate-B-surviving,
   closed-loop-supported). NOTE: the §7 re-attribution is NOT part of this — it was
   refuted; the honest statement is in the amended package §7.
2. **R-B:** retire gate (i) as measured evidence (algebraic proof supersedes; N-1).
3. **N-2 rulings R-0..R-5** (`CLAIM_N2_SELECTION_NUMERATOR_20260805.md.DRAFT` +
   `N2_SELECTION_NUMERATOR_DERIVATION_20260805.md.DRAFT`): R-0 the exoneration-scoping
   call (intake argued DISTINGUISHABLE with quotes); whether to adopt the S̄_φ-inside-1D
   formula in production (/physics-change, 5-item package drafted) given the measured
   bounded effect (+23–25 nats/h chord; does NOT un-rail 1D) and the M1-vs-live
   quadrature discrepancy (sign coherence 0.71) — or leave it as a documented bounded
   correction. The denominator question (matching D̃^φ change) is part of the same ruling.
4. **D1 disposition:** accept the bounded-null (tilt route) verdict; the simulation-side
   `ParameterSpace.p0` bounds retirement remains its own small /physics-change for future
   campaigns (unchanged from RUNBOOK-7 §1.2b; the 3135-event catalogue is still never
   re-scored band-blind).
5. **New intake: the f_k–pool coupling in L_cat** — route through /research-cycle stage 0
   (exoneration check first: "p_det inside/outside" ⚠ items are adjacent). This is now
   the top un-modeled-selection thread.
6. **Ledger numbering cleanup** — pre-existing July rows also numbered 90–94; the August
   thread (rows 90–94, 2026-08-04/05) collides. Author call: renumber July or fork a §1b.
7. **Calibration-gate next step:** extend `closed_loop_gfrac.py` toward the A3 criteria
   still open (realistic host-observation model / multi-candidate balls; σ–d_L joint
   texture; the P–P leg of §9's CONFIRM that was not evaluated) — then the stage-4 gate
   can adjudicate keep-digging vs report-bound for the 1D rail itself.
8. **Paper (#47)** remains ON HOLD; the calibration gate is closer but not yet the
   trusted-run gatekeeper RUNBOOK-7 §4 requires (P–P leg missing).

## 2. What ran where (provenance quick map)

| run | dirs (cluster $WS) | jobs | code |
|---|---|---|---|
| frozen-g | `run_20260804_frozeng_{iiib,joint_r1}` | 6148505/6148507 | 930a9484+c917ed87 |
| N-2 sel-1d | `run_20260805_n2sel1d_{iiib,joint_r1}` | 6152554–7 | 0167df53 |
| D1 A1/A2 | `run_20260805_d1_{a1,a2}_{iiib,joint_r1}` | 6152697–6152704 | 128f318a |
| closed-loop | local, `results/closed_loop_gfrac_20260805/` | — | 77b524af |

Local retrieved data (uncommitted, regenerable): `results/run_20260804_frozeng/`,
`results/run_20260805_n2sel1d/`, `results/run_20260805_d1/` (diagnostics + posteriors).
Readout scripts + JSONs are committed alongside each. Workspace expires **2026-09-23**.

## 3. Gotchas (new this session)

- **Agents disagree; adjudicate with an instrument.** Twice tonight two subagents
  produced conflicting numbers (g_frac near-scalar vs per-event; M1 point-eval vs live
  quadrature). The deciding move both times was a small deciding script committed as
  evidence (`adjudicate_g_frac.py`). Never present un-adjudicated agent numbers.
- **Class-summed statistics across venues need the paired/stratified read** (amendment
  A2) — the gate (vii) aggregate agreement was a two-effect cancellation; a conclusion
  built on it had to be withdrawn.
- **The diagnostics CSV answers most questions for free** (amendment A1) — frozen-g,
  the ablation, M1, M2, and both stratum reads all came from CSVs already on disk.
- `--out` on the F5 engine defaults to the paper's headline JSON — always redirect
  real_nz runs (already encoded in `docs/RESEARCH_CYCLE.md` stage 1).
- `.planning/` is gitignored — evidence files cited by ledgers/preregs must be
  force-added (`git add -f`), as done for the gfrac thread.
- Skill files with `disable-model-invocation: true` are unreachable without a CLAUDE.md
  trigger row (bit the research-cycle skill on day one; fixed).
- Counterfactual runs must set `--combine=false` era expectations: D1 arms emit NO
  combined posteriors by design — do not hunt for missing `combined_*.json`.

## 4. Standing constraints (unchanged, re-affirmed)

- The 3135-event catalogue stays band-passed; never re-score against band-blind objects.
- Counterfactual runs (frozen-g, sel-1d, D1 arms) are diagnostics, never results.
- Delivered-convention pins PRIMARY (D2) until the truth-convention promotion condition.
- C7-adjunct (Σ_glob smearing) LAST, alone, per binding ship order — untouched tonight.

## 5. Addendum (2026-08-05, post-wrap): the Hitchhiker independence thread

A same-day intake, run after the sections above were written. Full detail:
`results/campaign51_20260728/realistic_20260729/CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT`.

**Paper verification (P1/P2/P3).** The author recalled a claim from "The Hitchhiker's Guide to
Dark Siren Cosmology" (identified as arXiv:2212.08694, Gair et al., AJ 2023) and flagged it as a
possible conflation of two statements. Verification found **three** distinct statements, not two:
**P1** (§2.3, after Eq. 30) — the selection denominator's dependence on the latent galaxy field
`{z_g}` breaks separability unless galaxy redshifts are perfect, rescued only by a
**large-detection-volume / uniform-in-comoving-volume** argument, no shared host required; **P2**
(§2.3, Eq. 31, = the author's S1) — multi-event cross-terms, suppressed by `1/N_gal`, requiring
**both** imperfect redshifts **and** a host shared by ≥2 events; **P3** (§3.3, = the author's S2)
— the large-N_gal validity condition and its empirical one-galaxy-limit demonstration (bias HIGH
at δz/z = 3%, still present at 0.3%). **S1 and S2 are two ends of ONE statement** (P3's
large-N_gal condition is exactly what makes P2's `1/N_gal` suppression work) — the author was not
conflating unrelated things. The genuinely new find is **P1**, which the author's recollection did
not include and which needs no shared host at all — it is live wherever a selection object
depends on the latent galaxy field and is factored out under an assumption of large volume.

**M-1 verdict: KERNEL-FINITE**, and the idealized-venue scoping correction. `resolve_host_z_kernel`
(`bayesian_statistics.py:149-210`) has `volume_deconv` pinned explicitly in iiib's own
`run_metadata_0.json` (`host_z_kernel: "volume_deconv"`, `observed_catalogue: null`) — the
point/δ kernel branch is unreachable from that config. The per-galaxy width input is likewise
non-δ: the **parent** (22.6M-row) catalogue's `REDSHIFT_MEASUREMENT_ERROR` has minimum 5.24e-4,
median 0.0396, **0.0% exactly zero** (a parse-time peculiar-velocity floor,
`galaxy_catalogue/handler.py:434-479`, is folded into every row regardless of realization) —
median σ_z/(1+z) = **0.0330**, i.e. **0.94×** the GLADE photo-z scale (0.035) used at the joint
venue. **Flag for the record:** the parent catalogue's kernel width sitting this close to the
photo-z scale bears directly on the standing photo-z-starvation account (h0-railing root cause,
[[h0-railing-rootcause-photoz]]) — a kernel this wide at the "exact-z" venue means the
idealized/observed venue split is narrower than previously assumed, and any argument that leaned
on "idealized = exact z" to bound a correction's size should be re-examined at the kernel level,
not just the catalogue level. **Consequence:** the paper's perfect-z escape clause fails on every
venue in this repo; the claim draft's §5(a) refutation of H-1 (the 1D-rail-via-cross-terms route)
is **RESCINDED** (strikethrough + dated note in the draft). H-1 is **LIVE**. This is a scoping
correction, not a reversal of any *measured* result: nothing that was actually run changes; what
changes is that "idealized venue ⇒ exact host redshifts ⇒ this class of correction is zero" was
true of the data realization and false of the estimator kernel, and every earlier interpretation
that used that shorthand (including inside this repo's own prior citation of S1,
`docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:175-179`) needs that scoping applied before being reused.

**Author mandate (verbatim, binding on this thread and beyond):**

> "We need to measure the coupling terms suppressed by 1/N_gal to have an informed decision —
> either show it can be neglected or that it needs to be regarded. Do not refute it because it
> doesn't suit the current picture; that might be mathematically/scientifically just wrong."

**P1 → f_k convergence.** Open thread §4 item 15 (the `f_k` pool-fed completeness callable inside
`L_cat`, discovered via D1's N2-null failure, ledger row 94) now has a **literature warrant**: P1
(arXiv:2212.08694 §2.3, the paragraph after Eq. 30) is precisely the statement that a
selection/detection object depending on the latent galaxy field may only be factored out under a
large-detection-volume argument — and `f_k` is applied inside a *ball-sized* volume, where that
premise is exactly what does not hold. This does not resolve item 15; it gives it a citation and
raises its priority.

**Queue insertion — TOP item, next session, alongside the morning author queue (§1 above):**
build and pre-register the cross-term instrument specified in the claim draft's new
"The cross-term instrument" section — a per-galaxy-resolved re-evaluation of the candidate-ball
catalogue sums for the 385 overlap-involved events / 279 `d_L`-compatible shared-sky pairs
(C-4's census), computing the leading Eq. (31) cross-term at h ∈ {0.60, 0.73, 0.81, 0.86} minimum,
against a NEGLECT/REGARD negligibility band to be locked at pre-registration time from an
M-2-style cheap bound — not invented in advance. Ledger row 95 records the M-1 verdict and the
residual (`gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1).
