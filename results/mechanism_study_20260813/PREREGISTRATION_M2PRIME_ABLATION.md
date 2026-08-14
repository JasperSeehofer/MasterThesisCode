# PRE-REGISTRATION — A-M2′ term ablation + A-NULL specificity control (thread 17, stage 2)

**Date:** 2026-08-14 · **Status at authoring: DRAFT → REGISTERED at the commit that carries it** ·
**Authorized:** ledger row #102 item 6 ([DO], author 2026-08-14) · **Parent:**
`PREREGISTRATION_MECHANISM_ISOLATION.md` (`73141160`) — its §1 ladder, §3 anti-tuning clause, §5
validity checks and §7 closures are inherited verbatim unless explicitly superseded below.
**Discipline:** Amendment **A8 as revised and adopted** (ledger row #102 item 5; `docs/RESEARCH_CYCLE.md`
row A8): every branch names its satisfying arm and what it ablates; every rule naming a point
prediction is two-sided; **no count/class-based branch is adjudicated while a registered arm capable
of changing the classification is unrun**; every band states its derivation and false-fail rate.

**Toy-calibration bar (new, binding here):** per the commission review
(`results/commission_research_20260814/REPORT.md`, D1-03/D1-12) the M5 L0 toy is ruled unfaithful at
production K (ledger row #102 item 1). **No band or branch edge in this document is derived from any
L0 toy.** Every number below comes from committed instrument artifacts (campaign `d45fbf15`, arms
`9fd0386b`/`5b0bd17a`) or closed-form arithmetic shown in place.

---

## 1. The question

The completed stage-1 arms established an input condition (host-redshift exactness gates the bias)
and a shape (gate × amplifier), but no estimator term (row #102 item 2: branch 2 = PREMATURE
ADJUDICATION, A-M2′ unrun). **A-M2′ is the register's only estimator-side arm** and the destination
DS-M5's own refutation clause names. This registration runs it, together with the specificity
control the stage-1 design lacked.

### Candidate register state entering this stage (post row #102)

| id | term | status |
|---|---|---|
| **M2′** | missing measure/Jacobian **inside** the z-integral (`venue_transfer.py:1290-1296` at HEAD; the kernel-branch integrand `kern · p_gw` is integrated `dz` with no `|d d_L/dz|` factor; the σ_z = 0 point branch has no integral and hence no such factor to miss) | **OPEN — this study's target** |
| **M6** | σ_z-blind aggregate log-posterior **tilt** × dose-controlled **curvature** composite (registered §2 below) | OPEN — L0 obligations registered below; no instrument arm this stage |
| **M7** | host/impostor **ball-window inclusion asymmetry** (named in the intake dossier's parity text; assigned an M-ID by dossier Erratum E1) | OPEN — L0 formalization required before any arm |
| M1, M2, M3, M4 | — | CLOSED as per parent §7 (M1/M4 toy-independent; M3 on its analytic core, note: *plausible pending committed artifact*) |
| M5, M5′ | — | REFUTED as registered (DS-M5 instrument measurement); toy-dependent sub-closures NOT ESTABLISHED (row #102 item 1) — the instrument refutation stands |

## 2. The M2′/M6 tilt arithmetic — registered pre-data derivation (no toy input)

Write the kernel-branch candidate integral as coded:
`c₁ₖ(h) = ∫ N(z; z_obs,k, σ_k) · N(d_L(z,h)/d_obs; 1, σ_d) dz`. If the GW likelihood is a density
in its own datum, the change of variables to z carries the measure
`|d d_L/dz| = D′(z)/h` (with `d_L(z,h) = D(z)/h`). The missing factor decomposes:

- **ln J = ln D′(z) − ln h.** The **−ln h** piece is z-independent: on kernel-branch candidates it
  multiplies every c₁ₖ by 1/h, hence at full dose (all candidates kernel-branch) it adds exactly
  **−ln h per event** — a σ_z-blind tilt of slope **−N/h = −982/0.730 = −1345 nats per unit h** at
  truth. Its *absence* in the coded estimator is an **up-tilt of +1345 nats/h** relative to the
  measure-carrying form. The **ln D′(z)** piece is h-independent per candidate; it re-weights
  candidates *within* an event at O(σ⁰) — suppressed by D′'s flatness across the ball window, not
  by parity — and its residual h-coupling through the kernel is O(σ_z²) (parity-suppressed).
- Together with α's registered σ_z-blind up-tilt (+1.036·N/h = +1393 nats/h, parent §7/M4), the
  total σ_z-blind up-tilt is **≈ +2739 nats/h**, of which the missing-J share is **49.1 %**. The
  commission independently measured the aggregate gradient at truth as **2625–2720 nats/h,
  dose-invariant across cells**, with α-share 52.7 % predicted vs 53.3 % measured (REPORT.md,
  M6 kill tests). The prediction sits **1–4 % above** the measured band, not inside it —
  order-and-share consistency on committed data, quoted here as motivation, not as a band.
- **The M6 composite account** (registered): MAP displacement ≈ S_tilt · σ²_post, with S_tilt
  σ_z-blind and σ²_post dose-controlled. It predicts the host gate exactly (exact host ⇒ degenerate
  posterior ⇒ σ²_post → 0 ⇒ zero bias at any impostor dose), the non-additivity (curvature is
  jointly controlled), and passes the parity constraint (a multiplicative tilt is not a symmetric
  smoothing). **Registered point expectation for A-M2′** (WEAK, non-branch-carrying, two-sided):
  restoring J removes the 49.1 % tilt share, Δb ≈ −S_J·σ²_post = −1345 × (0.004386)² = **−0.0259**,
  i.e. b(A-M2′) ≈ **+0.011 ± 0.010** (consistency window; the ±0.010 spans the local-Gaussian
  approximation's demonstrated ~1.5× scale error in reproducing the null bias itself:
  2739 × 1.924e-5 = 0.0527 vs measured +0.0373). Values on **either side** outside this window are
  equally reportable misses of the expectation; **the branch reads DS-M1 classes only.**

### M6/M7 L0 obligations (registered; CPU-minutes, committed data only, before any new arm beyond §3)

- **M6-L0:** recompute the aggregate d(ln post)/dh at truth per cell from the raw committed
  `ln_post` vectors; registered kill tests: (i) tilt dose-invariant within ±10 % across all
  f_h > 0 cells; (ii) bias/σ²_post constant within a factor 2 across the 9 interior cells;
  (iii) α-share of the tilt = 52.7 % ± 5 pp. Any failure kills the composite as stated.
- **M7-L0:** formalize the inclusion asymmetry (ball membership decided on true z, kernels read at
  scattered z; host is the one candidate whose membership and datum coincide) and derive its sign
  and σ_z-order; if the leading effect is O(σ_z²) it is parity-killed before any arm.

## 3. The arms — exactly the two unspent L1 slots (parent budget: 3 of ≤ 5 used)

| arm | what changes (estimator-side ONLY; generator untouched) | seeds | N | prediction |
|---|---|---|---|---|
| **A-M2′** | the measure factor `J(z,h) = D′(z)/(h·d_obs)` restored inside the kernel-branch integrand (both channels; point branch untouched) | **+53000…+53024** (fresh, disjoint — checked by unit test before any run) | 25 | DS-M1 class read; weak expectation §2 |
| **A-NULL** | kernel-branch integrand multiplied by the constant **1.7** (z- and h-independent) | **+50000…+50014, PAIRED** — the same seeds as MN0X's first 15 (deliberate paired-determinism design, registered here, not post-hoc: the prediction is exact equality, not re-estimation) | 15 | per-seed MAP grid indices **exactly equal** to MN0X's stored records; ln_post shifted by exactly N·ln 1.7 within rtol ≤ 1e-12 |

**A-NULL inertness derivation (registered):** at full dose every candidate row takes the kernel
branch (GLADE-empirical σ_z sampler has min σ_z = 5.3e-4 > 0 on the pruned frame), so the ×1.7
scales every c₁ₖ and c₂ₖ uniformly ⇒ L_i → 1.7·L_i ⇒ ln-posterior shifted by +N ln 1.7 at every h ⇒
argmax invariant. **False-fail rate: bounded by the probability that two stored per-seed grid values
sit within the FP propagation error of one extra rounding per candidate (≤ ~1e-12 relative); the
committed MN0X margins between adjacent grid log-posteriors are O(1–10³) nats ⇒ false-fail ≈ 0.**
A-NULL failing is therefore evidence about the machinery, not noise — hence its STOP role below.

**A-M2′ code form:** `J = |d d_L/dz| / d_obs` evaluated at the quadrature nodes by **central
difference of `dist_vectorized` with registered step ε = 1e-6 in z** (deterministic, no RNG;
`physical_relations.dist_derivative` exists but is scalar-only with no array fast path, so the
vectorized central difference is the registered form — noted here at drafting, before
registration); multiplied into `integ` (hence into both c₁ and c₂) on kernel-branch rows only. The exact diff is fixed in `ARMS.md` (appended, registered with this file)
before any run. At σ_z = 0 the arm is **identically** the base estimator (point branch untouched) —
constraint (a) preserved by construction, unit-tested.

**Costs at the realized 0.969 CPU-h/seed anchor:** A-M2′ ≈ 24 CPU-h; A-NULL ≈ 15 CPU-h. L2 remains
≤ 1 arm, author-order only, none requested.

## 4. Decision statistics — bands locked at this commit (derivations in place)

- **DS-M1 (headline, 1D; 2D alongside)** — edges carried verbatim from the parent §3, applied at
  N = 25 (SE ≈ 0.005/√25 = 0.0010): TERM-OWNS |b| ≤ 0.010 ∧ HPD90 ≥ 0.60 · TERM-PARTIAL
  0.010 < |b| < 0.030 · TERM-INNOCENT |b| ≥ 0.030 ∧ |b − b_ref| ≤ 0.004 · OTHER = anything else.
  **b_ref = +0.037250 (MN0X, N = 100, committed)** — the row-#102 discharge of the null reference
  (item 3). Separations between class edges are ≥ 10σ at N = 25.
- **DS-N1 (A-NULL)** — PASS iff (i) all 15 per-seed 1D and 2D MAP grid indices equal MN0X's stored
  indices AND (ii) the **floor-aware integer shift law** holds at every h in both channels: with
  Δ(h) = ln_post_ANULL(h) − ln_post_MN0X(h) and m(h) = round(Δ(h)/ln 1.7), require
  |Δ(h) − m(h)·ln 1.7| ≤ 1e-6 nats and 0 ≤ m(h) ≤ N. **Derivation:** an event floored at the
  registered −745 zero-event penalty (`_LN_ZERO_EVENT`) has L_i = 0, which is invariant under ×1.7,
  so the exact shift at any h is (N − n_floored(h))·ln 1.7 — the committed MN0X records **already
  show floors firing at off-peak h in ≥ 7 of the 15 paired seeds** (jumps of ~727–842 nats), so a
  naive "N·ln 1.7 everywhere" rule would deterministically false-FAIL on sound machinery; the
  integer law is exact modulo FP rounding (≲ 1e-8 nats accumulated; tolerance 1e-6 gives ≥ 100×
  headroom). m(h) = N wherever no floor is active, including each seed's MAP neighbourhood when
  floor-free — reported, not required, since floors at isolated grid points near the plateau are
  present in the stored data. **The decision weight is carried by (i)**, whose false-fail is
  genuinely ≈ 0: the measured minimum top-2 grid gap across the 15 stored seeds is 0.048 nats (1D)
  / 0.030 nats (2D), ≫ the ~1e-8-nat FP scale of one extra multiply. Two-sided by construction
  (equality). **FAIL of (i), or of (ii)'s integer law, ⇒ abort (d′) below.**
- **DS-M6-L0 / DS-M7-L0** — the §2 kill tests, each two-sided as written.
- **Anti-tuning:** every number above is fixed at this commit, derived from committed artifacts or
  closed-form arithmetic shown here; none may be adjusted after any arm runs.

## 5. Branches (A8-v2 form; presented to the author, never self-adjudicated)

**Execution-completeness clause (BLOCKING):** no branch below is adjudicated until **both** A-M2′
and A-NULL have run, or the unrun arm is withdrawn by an author [RULE].

**Split-precedence clause:** branches 2–4 each require their named DS-M1 class in **both channels**;
**any 1D/2D class split routes to branch 5**, which takes precedence over branches 2–4.

1. **STUDY-CONFOUNDED** — satisfying arms: **A-NULL** via DS-N1 (ablates a provably-inert constant
   factor), or **either arm** via a §6 validity failure. Fires iff DS-N1 FAILS or any §6 validity
   check fails. Meaning: the instrument or harness is unsound; every measurement in this stage is
   void; author call on repair-and-rerun.
2. **M2′-OWNS** — satisfying arm: **A-M2′** (ablates the z-integral measure). Fires iff A-M2′ is
   TERM-OWNS (both channels). Meaning: M2′ is the identified mechanism; the `/physics-change`
   new-formula slot is written against J-restoration with this arm as its regression test
   (author-gated as ever).
3. **M2′-PARTIAL** — satisfying arm: **A-M2′**. Fires iff TERM-PARTIAL in **both channels**.
   Meaning: M2′ contributes but does not own; the registered follow-up is the M6 composite
   decomposition (§2), starting from its L0 obligations — **no repair is proposed from a partial
   read.**
4. **M2′-INNOCENT** — satisfying arm: **A-M2′**. Fires iff TERM-INNOCENT (both channels). Meaning:
   the estimator-term register is exhausted **as of this stage** ({M6, M7} remain, neither is a
   single-term ablation); the parent's NO-OWNER handling binds: **mandatory Stage-L literature
   sweep before any further arm.**
5. **OTHER / SPLIT** — any remaining outcome (incl. a 1D/2D class split, which the parent §6 marks
   as itself a finding). Reported raw, direction stated, no branch forced.

## 6. Validity and STOP criteria

- **V-M2/V-M3/V-M4** carried verbatim from the parent §5 (generator invariance incl. AR-1..AR-3;
  pin integrity; clean rule). Both arms consume the identical pre-dose realisation discipline;
  A-M2′/A-NULL differ from the base **only** in the estimator switch.
- **V-M5** — values golden at the running HEAD against the committed MN0X records, rtol ≤ 1e-12,
  MAPs exactly equal, re-executed before any arm (the D1-13 independent re-execution, closing it).
- **Abort:** (a) non-finite ln_post > 1 % ⇒ STOP; (b) horizon-drop > 5 % ⇒ STOP; (c) any V-M
  failure ⇒ STOP; **(d′) DS-N1 FAIL ⇒ STOP** (supersedes the parent's toy-referenced abort (d) for
  this stage — no toy participates in any rule here, per the toy-calibration bar).

## 7. Expected NULLs, pre-registered

- **A-NULL:** exact MAP equality (that is its PASS). Anything else is not noise (§4).
- **A-M2′ landing TERM-INNOCENT** would refute the §2 weak expectation and kill the J-half of the
  tilt account while leaving the M6 composite (α-half + curvature) alive — that combination is
  informative, not contradictory, and routes through branch 4.
- **A-M2′ landing TERM-OWNS** would exceed the §2 expectation (predicted PARTIAL); the M6
  decomposition would then need the α-share re-derived — recorded now so the surprise is legible.
- **2D tracks 1D** in both arms; a split forces branch 5.

## 8. Provenance

Parent prereg `73141160` · ledger row #102 (authorization + rulings) · commission
`results/commission_research_20260814/REPORT.md` (adopted; toy bar D1-03/D1-12; M6/M7 origin) ·
committed references: MN0X records `5b0bd17a`, campaign decision cell `d45fbf15`, realized cost
anchor 0.969 CPU-h/seed (runbook §4) · instrument at the registering commit + the `ARMS.md`-fixed
switch diffs. Registered documents are append-only from the registering commit onward.
