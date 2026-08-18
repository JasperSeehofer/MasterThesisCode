# PRE-REGISTRATION — G-2: spec-z-kernel σ_z → 0 cells ([C-SYM]/[P3] front, harness scale)

**Date:** 2026-08-18 · **Status:** DRAFT v1 — awaiting verifier pre-check. **Execution
authorization:** the author's 2026-08-18 in-session directive (verbatim: *"Please move through
this research cycle autonomously and flag it accordingly. Make sure you dont make the same
mistake as last cycle where one of the instrument arms was void in the end. looking forward to
see the results!"*) grants autonomous execution of this cycle's [DO]-class steps
(orchestrator-derived reading, flagged for veto: freeze, build, preflight, run, readout);
**every branch call below still returns to the author as a [RULE], presented not adjudicated.**
**Append-only after commit: no edits above the VERDICT line once committed. Any later edit to
§7 voids the registration, except the single registered pretuning fill-in, itself append-only.**

**Provenance gating:** row #121 item 3 (grounding-measurement collection directive); proposal
§6 row G-2 (`docs/PROPOSAL_GRAY_CONVENTION_PAPER_INTEGRATION_20260817.md`): the paper's σ_z → 0
no-cost limit is an **orchestrator-derived, unmeasured structural argument** (§1.2, UNVERIFIED
item i) — decision item 3 asks whether the paper may state it caveated or must measure it. The
Stage-L R0 sweep (2026-08-18) confirmed **no cited paper states a σ_z→0 validity condition for
this bias class** — the claim is repo-internal and G-2 is its only possible ground.
**Distinct from the exonerated spec-z-subset rescue (#42):** that was a production H0-rescue
claim on GLADE spec-z hosts; G-2 is a harness-venue limit measurement of the insertion
mechanism's σ_z dependence. No rescue is claimed on any branch.

**The honest two-sided physics (registered before running):** the §1.2 argument — "the omitted
weight acts through the numerator's z-marginalization over the photo-z kernel; σ_z → 0
collapses the kernel and the weight normalizes out" — applies structurally to the
**catalogue-leg** insertion (per-candidate photo-z kernels). The **completion-leg** insertion
acts through a z-integral whose width is set by the GW distance likelihood and the completion
window, NOT by σ_z — so its displacement is predicted to PERSIST as σ_z → 0. G-2 therefore
separates the legs and registers a hypothesis per leg; **either outcome is decision-grade**:
collapse-confirmed scopes the paper's wording to the catalogue leg; persistence-refuted forces
the wording to drop the general "no-cost at spec-z" framing entirely.

## 0. Instrument

The [A3] harness plus the **G-1 selection_cell extension** (`cat1d`; spec, tests t1–t5, and
instrumentation status in `PREREGISTRATION_G1_CATLEG_SYMMETRY.md` §0 — shared freeze commit).
G-2 additionally uses the existing `1d` mode (completion-leg-only [P2]) and the existing
`--sigma-z` / `--n-z-quad` knobs (`pp_coverage.py:620,2609,2686-2691`). **Instrument limit
(registered):** σ_z = 0 exactly is a singularity (`_norm_pdf` divides by σ; the smeared
catalogue density uses dz = σ_z/16) — the smallest rung is a finite spec-z-class value and the
σ_z → 0 statement is an extrapolation from the measured trend, stated as such.

σ_z is **generative AND kernel width jointly** per rung (the harness venue is self-consistent
at every rung); each read is convention-vs-convention WITHIN a self-consistent venue — the
right object for the claim, registered so the venue-change-per-rung is not misread as a
confound.

## 1. Hypotheses (registered BEFORE any cell is run)

Let C_cat(σ_z) = paired per-realization MAP delta (cat1d − off) and C_comp(σ_z) = paired delta
(1d − off), at V-deep, per truth; primary truth h = 0.72 (the amplitude truth of record).

- **H-Zcat (the σ_z→0 no-cost limit, catalogue leg).** |C_cat| collapses: at the smallest rung
  it is null (§4 two-sided band) and the rung sequence is monotone non-increasing.
  PASS-meaning: the paper may state the no-cost limit for the catalogue-leg weight as a
  measured statement.
- **H-Zcomp (completion-leg persistence — the registered counter-prediction).** |C_comp|
  persists: at the smallest rung it retains ≥ ⅔ of its σ_z = 0.035 amplitude with the same
  sign. PASS-meaning: the completion-term cost is NOT a photo-z artifact — the paper's σ_z → 0
  wording must be scoped to the catalogue leg only. FAIL-meaning (C_comp collapses too): the
  entire insertion phenomenology is photo-z-driven; the V-deep boundary itself is
  σ_z-conditional and [C-SYM]'s scoping inherits that (returns to stage 0 with the data).
- **Engagement preconditions (registered):** |C_cat(0.035)| ≥ 0.010 AND |C_comp(0.035)| ≥
  0.010 at h = 0.72 at the registered Q* quadrature. A leg failing its precondition ⇒ that
  leg's limit read is **UNDETERMINED-BY-DESIGN** (no amplitude to collapse), reported as such,
  never scored PASS/FAIL. (Anchors: C_comp ≈ −0.033 [registered, prodcal S-1]; C_cat ≈ +0.041
  [ad-hoc audit, UNVERIFIED — G-1's H-CAT registers it first].)
- **Expected NULLs:**
  - **N-a (engagement at rung 1):** both registered pairs non-degenerate at σ_z = 0.035.
    (At smaller rungs a degenerate C_cat is a legitimate physical outcome, NOT a null
    violation — registered so it cannot be misread as instrument failure.)
  - **N-b (off-cell continuity):** the rung-1 `off` cell's absolute bias per truth lies within
    3·combined-SE of the prodcal `vdeep_250_production_off` cell (different seed ⇒ class
    comparison). Outside ⇒ Q* changed the venue: STOP, audit quadrature before any read.
  - **N-c (quadrature convergence, scored gate):** at the smallest rung, raising Q* to the
    smallest registered sweep value ≥ 2·Q* (160→480, 240→480, 480→960) on an R=8 probe (seed
    20280399, probe-flagged, archived under `preflight/`) changes the per-truth MAP bias by
    ≤ 0.0005. Violation ⇒ STOP (instrument not converged; return with a raised-Q* amendment).
    *(AMENDMENT G2-4, applied verbatim.)*

## 2. Stage-1 information forecast / power

- Paired-delta SE classes at n=250, R=120, shared stream (measured, prodcal record): the
  flat-venue near-null class is 1.4–1.7e-4 (V-prod/V-flat pairs); **at V-deep — this
  registration's venue — the engaged-lever class is 7.1–8.1e-4** (fused−off), and the paired
  SE shrinks with the lever, so collapsed-rung pairs are expected near the near-null class.
  Dynamic range: C_comp baseline ≈ 0.031 (paired 1d−off; the −0.033 figure is the `1d` cell's
  absolute bias) ⇒ the ⅔-persistence edge (0.021) vs the 0.10-collapse edge (0.0031) are
  separated by ~23 realized paired-SE — decisively powered; H-Zcat at the precondition minimum
  (baseline 0.010) has edges separated by 0.0023 ≈ 3 realized SE at the rung-1 class — the §4
  overlap guard owns the residual risk, and the separation is ≥ 14 SE if the rung-3 pair
  lands in the near-null SE class as expected. *(AMENDMENT G2-2, applied verbatim.)*
- The binding uncertainty is **instrumental, not statistical**: quadrature fidelity at small
  σ_z (N-c gates it) and the runtime multiplier κ of the raised Q* (pretuning measures it;
  ceiling binds).
- False-fail (A8-v2 disclosure): H-Zcat's two-sided null edge max(0.10·|C_cat(0.035)|,
  2·paired-SE) ≥ 2σ by construction ⇒ ≤ 5% at the single primary truth; monotonicity is
  checked with a 2·SE tolerance per step (compounded false-fail ≤ ~10%, accepted and
  disclosed). H-Zcomp edges sit ≥ 45 SE from the measured baseline — false-fail negligible.
- No Fisher leg (not a repo asset).

## 3. Design — cells, rungs, seeds, scorer

**Venue:** V-deep exactly as G-1 §3 (z_support 0.40, sky_frac 1e-4, default d50, n_galaxies
2e5, α_M 0.25, catalogue_mode, production noise).

**Rungs:** σ_z ∈ {0.035, 0.010, 0.002} (rung 1 = the production-class scatter of record;
rung 3 = spec-z class, safely above the dz = σ_z/16 instrument floor).

**Cells:** {off, 1d, cat1d} × 3 rungs = **9 cells**, each n_events = 250, R = 120, truths
{0.62, 0.72, 0.84}, h_step 0.004, **one master seed 20280311 shared across ALL 9 cells** (the
generative stream is σ_z-continuous, keeping cross-rung latent draws aligned; within-rung pairs
are exactly stream-paired). **n_z_quad = Q\*** fixed across ALL cells (a single instrument for
the whole grid — the only axis varying between rungs is σ_z itself). Seed freshness (verified against the full extracted `"seed"` inventory of `results/`,
2026-08-18): 20280311 and 20280399 collide with nothing. Consumed families: prodcal
20270818–20271333; coverage 20260701–20261207; calibration-gate/venue-transfer harnesses
20260805–20261004 **and 20280808–20306007** (the nearest consumed seed, 20280808, is 497
above 20280311; a different instrument in any case); G-1's deliberate reuses.
*(AMENDMENT G2-3, applied verbatim.)*

**Registered pretuning fill-in (§7; archived, never scored):** seed 20280399, σ_z = 0.002,
n = 250, R = 8: sweep n_z_quad ascending over {default, 240, 480, 960}; **Q\*** = the smallest
value whose per-truth MAP bias changes by ≤ 0.0005 vs the next larger value; record the
runtime multiplier κ vs the default. Exhaustion (960 not converged) ⇒ STOP → author →
amendment. Only convergence and runtime fields are consulted.

**Registered pairs (PAIRS manifest, scorer `readout_g2.py` pre-committed):** per rung r:
Pc(r) = cat1d(r) − off(r), Pd(r) = 1d(r) − off(r) — 6 pairs. [A2] paired distributions
reported alongside class means, always. No statistic outside the scorer enters the verdict.

**Budget:** per-rung cost at κ = 1 (measured prodcal classes): off 416 s + 1d ≈ 500 s + cat1d
≈ 1000 s ≈ 0.53 CPU-h ⇒ 1.6 CPU-h × κ for the grid; pretune ≤ 0.5. **Ceiling 6 CPU-h**; if
pretuned κ projects the grid above the ceiling ⇒ STOP → author (do not trim rungs or cells
unilaterally). Local dev machine, single core per cell. **Disclosure (AMENDMENT G2-6):** at
the plausible Q* ∈ {480, 960} the z-quadrature-dominated cell cost scales ~×3–6, projecting
the grid at ~4.8–9.6 CPU-h against the 6 CPU-h ceiling — the registered κ-STOP is *likely* to
fire at the first pretune readout and its firing is the designed author-return, not an
anomaly. Also registered: the N-c convergence evidence covers `n_z_quad` only; the fixed
internal grids (`_posterior_normalizers` ngrid = 3000 → spacing 3.2e-4 ≈ σ_z/6 at rung 3;
D(h)/β_G at 3000 nodes; `n_z_survival` = 1500 on the smooth σ_z-independent S̄ table) are
adequate by spacing argument, not by the doubling probe.

## 3b. Arm-validity preflight (registered, non-scored — the anti-void gate)

Author-mandated this session (the V-ctrl lesson: an arm ran 120 realizations with
completion_fraction ≡ 0 because no one checked engagement before running). **Before any scored
realization, every registered cell runs an R = 4 probe at its exact registered configuration**
(same seed base, probe-flagged) and must show:

1. completion_fraction ∈ [0.05, 0.95] and catalogue-bearing event fraction > 0.3;
2. every registered pair engaged at rung 1 / in-cell: the probe's paired delta distribution
   non-degenerate wherever §1 expects engagement;
3. finite likelihoods (no NaN/rail-pinned probe MAPs at every truth);
4. S̄_φ table support covers the venue's completion window AND the catalogue candidates' kernel
   range (the cat1d lever has something to act on).

Any violation ⇒ **STOP before the scored run**, diagnose, return as an amendment. Probe
outputs archived in `preflight/`, never scored, never quoted as measurements.

## 4. Falsifiable bands (registered)

Primary truth h = 0.72; other truths reported, sign-coherence noted.

| read | PASS | FAIL | MIXED (first-class) |
|---|---|---|---|
| **H-Zcat**, Pc(rungs) | \|C_cat(0.002)\| ≤ max(0.10·\|C_cat(0.035)\|, 2·paired-SE) AND \|C_cat\| monotone non-increasing across rungs (2·SE tolerance per step). **Overlap guard (registered, AMENDMENT G2-1):** if 2·paired-SE(rung 0.002) ≥ ⅓·\|C_cat(0.035)\| the PASS and FAIL windows intersect — the H-Zcat read is then UNDETERMINED-BY-NOISE (unscored; designated next measurement: the same rung-3 pair at n = 800, returns as an amendment), and neither PASS nor FAIL is adjudicated | \|C_cat(0.002)\| ≥ ⅓·\|C_cat(0.035)\|, same sign (limit REFUTED for the catalogue leg) | else (partial collapse: report the measured fraction; the paper wording becomes quantitative, not asymptotic) |
| **H-Zcomp**, Pd(rungs) | \|C_comp(0.002)\| ≥ ⅔·\|C_comp(0.035)\|, same sign (persistence confirmed) | \|C_comp(0.002)\| ≤ 0.10·\|C_comp(0.035)\| (completion cost collapses too) | else (partial persistence: measured fraction reported) |
| preconditions | both legs ≥ 0.010 at rung 1 | — | a failing leg ⇒ that leg UNDETERMINED-BY-DESIGN (unscored, registered §1) |
| **N-b** off continuity | within 3·combined-SE of prodcal off, per truth | outside (STOP: quadrature changed the venue) | — |
| **N-c** quadrature | ≤ 0.0005 MAP-bias shift on Q* doubling probe | > 0.0005 (STOP: not converged) | — |

**Branch-referent check (A8-v2):** every branch above names its cells (Pc/Pd at registered
rungs); no branch's meaning exceeds what the 9-cell grid can establish. **Two-sidedness:**
H-Zcat's point prediction (0 at the smallest rung) carries a two-sided null band; H-Zcomp's
point prediction (persistence at ~full amplitude) is bounded on both sides by the ⅔ and 0.10
edges with MIXED between. **Execution-completeness:** no branch is adjudicated before all 9
cells and both probes complete (or a STOP fires first).

**Branch calls (each returns to the author as a [RULE] with the data):**
- H-Zcat PASS + H-Zcomp PASS: proposed wording of record — "the catalogue-leg selection
  weight's cost vanishes at spectroscopic precision (measured); the completion-term cost is a
  population-integral property, persisting to spectroscopic precision (σ_z = 0.002, measured)
  and stated as a trend extrapolation below that (caveat 1)" *(AMENDMENT G2-5)* — decision
  item 3 answered measured-first,
  superseding the caveated-statement option.
- H-Zcat FAIL: the no-cost limit is refuted even for the catalogue leg — the paper must drop
  the σ_z → 0 framing; [C-SYM]/paper scope returns to the author.
- H-Zcomp FAIL: the σ_z-conditionality generalizes; the V-deep asymmetric-insertion boundary
  (rows #120–#124) is itself σ_z-conditional — flagged back to the [C-SYM] claim card as new
  stage-0 evidence.
- Any STOP: returns with diagnosis and a proposed amendment; no unilateral design change.

## 5. Materiality yardstick (registered)

Same as G-1 §5 (⅓ × 0.0053 = 0.0018 in h) for any residual small-σ_z cost quoted to the paper.

## 6. Carried caveats and validity limits (registered)

1. σ_z = 0.002 stands in for σ_z → 0; exact zero is an instrument singularity (§0); the
   asymptotic statement is an extrapolation of a 3-rung trend, and is worded as such on every
   branch.
2. Venue: V-deep only (the amplitude venue). The production regime's small-lever behaviour is
   already covered by the three lever-closure measurements (row #124); no flat-S̄ σ_z rung is
   run here (the raised-d50 confound, VERDICT-3, would contaminate it).
3. Rung-1 C_cat anchor is UNVERIFIED until G-1's H-CAT registers it; if G-1 and G-2 both run,
   the G-1 value at σ_z = 0.035 (same venue, different seed) is the cross-check.
4. Venue-scoping binds both ways; no production change on any branch.
5. 1D channel carries the read (S-1: 2D inert at this venue); 2D deltas reported descriptively.

## 7. Execution appendix — filled at freeze; pretuning fill-in appended after Q* lands

- **Instrument hash of record:** shared with G-1 (single freeze commit; that commit is the
  hash of record).
- **Invocations of record:** pretune via `uv run python pretune_g2.py` (wide grid per A-PF-4;
  writes `pretune_g2.json` with Q*, κ, per-step convergence deltas); preflight via
  `uv run python run_g2.py --preflight` (executed pre-freeze on the wide grid: READY — the
  narrow-grid cat1d rail at h_true=0.84 relaxed 1.0 → 0.25); cells via
  `uv run python run_g2.py` (refuses a pretune not certified on the wide grid); scoring via
  `uv run python readout_g2.py --registered cells/` → `readout_g2_output.json`.
- **Environment:** local dev machine, single core per cell, sequential.
- **Pretuning fill-in appends below (append-only):** Q*, κ, convergence table, projected grid
  cost vs the 6 CPU-h ceiling (κ-STOP → author if exceeded, per §3/G2-6).

---

## PRE-FREEZE AMENDMENT A — 2026-08-18 — grid headroom (responsive to the registered §3b
## preflight STOP; applied before freeze, mirrors G-1 PRE-FREEZE AMENDMENT A)

The §3b preflight fired a genuine STOP: `rung_0.035_cat1d` rails at h_true = 0.84 (R=4 probe,
all MAPs at the grid edge h_max = 0.86) — mechanism-consistent (+0.04-class positive
catalogue-leg displacement pushes the top truth off-grid), i.e. a grid-headroom void arm
caught pre-run. Fix: **all 9 registered cells run on the extended estimator grid
h ∈ [0.56, 0.92]**, h_step 0.004 unchanged, whole-step extension (interior points aligned).
The preflight re-runs on the wide grid and must clear before any scored cell. N-b continuity
(off rung-1 vs the prodcal off cell) is retained as registered: interior grid NODES align, but
interior logL values shift slightly on the wide grid (the per-event z-quadrature windows
derive from h_grid.min()/max(), so they widen ~±7% and coarsen ~12% at fixed n_z_quad) —
shifts far below N-b's 3·combined-SE tolerance, which is why the class comparison stands; a
railed prodcal comparison truth is excluded from N-b, flagged. *(A-PF-3, applied verbatim.)* **Rail-fraction validity
gate (registered, mirrors G-1 item 6):** any cell × truth with 1D rail fraction > 0.10 on the
wide grid is UNDETERMINED-BY-RAIL for that truth's reads. **Precedence (registered, A-PF-1):**
band legs quantified "at every truth" are evaluated over the non-rail-flagged truths only, and
any PASS/FAIL adjudication requires ≥ 2 scoreable truths; a read with ≥ 2 truths
UNDETERMINED-BY-RAIL is itself UNDETERMINED-BY-RAIL (unscored, returns to the author with the
rail diagnostics). A rail-flagged truth never counts toward a "coherent at ≥ 2 truths" FAIL
leg. Budget ≤ ×1.38 (node ratio 91/66; ×1.29 estimated, A-PF-2) ⇒ grid ≈ 2.1–2.2 CPU-h × κ at
κ = 1; ceiling 6 CPU-h and the κ-STOP unchanged. **A-PF-4 (registered):** the pretuning sweep
(Q* selection, seed 20280399) and the N-c doubling probe run **on the wide grid [0.56, 0.92]**
— the grid whose z-window coarsening the convergence gate must bound; any Q* evidence produced
on the narrow grid is void for scoring.

---

## VERDICT

*(append-only below this line after execution)*
