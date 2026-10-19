# [HIER] Blocker analysis — control (PA-HIER-3), prior (PA-HIER-6), identifiability statistic (PA-HIER-7), wiring test (PA-HIER-16)

**Date:** 2026-08-27 · **Author:** subagent, zero-compute code-reading analysis only ·
**Scope:** turn each of the four named blockers into a decision the author can make in one line,
or resolve it outright where it is fact rather than judgement. **This document does not rule on
anything.** All file:line citations below were read directly from source during this pass.

---

## 1. CONTROL (PA-HIER-3)

### 1.1 What the code actually does (re-verified independently)

`realize_observed_catalogue` (`darksiren_emri/galaxy_catalogue/observed_realization.py`, NOT
`validation/` — the task brief's path is stale; the module lives under `galaxy_catalogue/`):

- `z_obs = z_g + sigma_scale * z_error_g * N(0,1)`, floor-clipped (`:331-334`).
- The **z_error column is never rewritten**. In `_realize_and_write`'s output-write block
  (`:454-462`) only `z_col`, `mstar_col`, and `mstar_err_col` are replaced; `z_err_col` is absent
  from that block — it round-trips as the parent's original string, confirmed by the docstring
  itself (`:185-187`, `:449-453`): *"the z width law is scale-free in z, so the stored column IS
  the width the kernel consumes and `sigma_kernel == sigma_realized` identically"*. This is a
  **deliberate, ratified design property of the realistic-host-observation-model gate**
  ([RATIFY-R2], `:11-14`), not a bug — it exists for a *different* purpose (a faithful
  observed-vs-true catalogue pair) than the one S0-R tries to press it into (an
  estimator/generator width-mismatch positive control).
- `host_pool_for_sigma_scale` (`correspondence_1d.py:1850-1891`) loads the **realized** CSV as an
  ordinary catalogue (`:1889-1891`) and returns ONE `GalaxyCatalogueHandler` that is used for BOTH
  (a) the mirror-universe host draw / `z_true` draw inside `draw_realization`, and (b) — per its
  own docstring, `:1868-1871` — direct reuse as `BayesianStatistics.evaluate`'s `galaxy_catalog`
  argument, i.e. the **estimator**. Confirmed at the call site: `run_mirror_seed_inprocess`
  (`:2683-2857`) takes a single `galaxy_catalog: GalaxyCatalogueHandler` and passes it straight
  into `bs.evaluate(galaxy_catalog, ...)` (`:2844-2857`) — literally the same object, same
  z/z_error columns, that fed the generator.

**Consequence, independently re-derived, matching PA-HIER-3's finding exactly:** at every
`sigma_scale`, the value that generated `z_obs` (`z_error` column) and the value the estimator's
kernel reads as its quoted width (`host_z_error` at `bayesian_statistics.py:6223-6224` /
`:6878-6879`, sourced from the SAME catalogue's `z_error` column) are **the identical number**.
`sigma_scale` perturbs *which galaxy sits at which z* (a population/pairing perturbation); it
never creates a width **mismatch** between the process that produced the data and the process that
scores it. S0-R as constructed cannot inject a z-kernel misspecification at any dose, 1.5 included.

### 1.2 A trap to flag explicitly: the naive "fix" collides with PA-HIER-1

A tempting shortcut — feed the estimator the **unscattered parent** catalogue (`sigma_scale=0`,
i.e. `z = z_true` exactly) while generating events from the **scattered** realization — does not
inject an `s`-axis misspecification either. It reproduces the *already-flagged* PA-HIER-1 defect
(`host_mode="catalogue"`'s generating law is a delta at `z_g`; feeding the estimator a
zero-scatter catalogue as its assumed center while it still integrates a finite-width kernel around
that center is exactly the `s → 0` truth-θ confound PA-HIER-1 already found, not a controlled
`s = 1.5` injection). Any control design must be checked against this collision before it is
trusted.

### 1.3 What a genuine control requires

The estimator's assumed width (the number in the `z_error` column it reads) must be **numerically
different from** the width that actually generated the realized `z_obs` values, while both sides
still center on the same observed position `z_obs` (an observer never sees `z_true`; only the
*quoted uncertainty* should be wrong, not the observation itself — otherwise the injection
conflates a `b`-axis and `s`-axis defect, exactly the S0-R "joint z+mass" problem PA-HIER-3/§5.2
item 8 already register for the mass column, now for z too).

**Concrete design (matches the recon's option (a), fleshed out):**

1. **Generator-facing catalogue** (`gen.csv`): the existing `realize_observed_catalogue(...,
   sigma_scale=s_gen)` call, unmodified — `z_obs = z_true + s_gen · z_error · N(0,1)`. This is the
   catalogue that feeds `host_pool_for_sigma_scale`'s host draw / `z_true` draw
   (`draw_realization`) — i.e. it determines what data actually exist.
2. **Estimator-facing catalogue** (`est.csv`, NEW): byte-copy every column of `gen.csv` except
   `z_error`, and write `z_error_est = z_error_gen / s_gen` into that one column. Same host
   identities, same observed z positions, same sky/mass — only the *quoted* width differs.
3. **NEW harness plumbing** (the substantive part): `host_pool_for_sigma_scale` /
   `run_mirror_seed_inprocess` currently thread **one** handler through both roles (§1.1 above).
   A control run needs **two** handlers — `gen.csv`'s handler feeds `draw_realization`;
   `est.csv`'s handler is what gets passed as `galaxy_catalog` to `bs.evaluate`. This is a new
   fork in the driver, not a parameter change to an existing function.

At truth θ = (0, 1) evaluated against `est.csv`, the estimator's assumed width is
`z_error_gen / s_gen`, while the true generating width is `z_error_gen` — a genuine ratio of
`s_gen`. The registered θ-grid's `s` hook (scaling `est.csv`'s quoted column) should then recover
`ŝ ≈ s_gen` at its peak, exactly the S0-R prediction as originally intended.

An algebraically equivalent alternative (recon's option (b) — scale the generating noise up and
leave the quoted column alone) requires the identical new plumbing (two catalogues, decoupled
generator/estimator roles); it is not cheaper, only a different arithmetic placement of the same
factor. Neither option is achievable by an existing flag.

### 1.4 Existing-flag search (exhaustive within the CLI/harness surface read)

Grepped `arguments.py`, `correspondence_1d.py`, `observed_realization.py` for any knob that
independently touches the z_error *column* apart from the z *draw*: none exists.
`--smear_global_selection` / `smear_sigma_z` (`bayesian_statistics.py:2664`,
`arguments.py:773-786`) only toggles whether the **global selection denominator** (site 2.3, a
single per-h scalar) integrates over a kernel at all — it is an estimator-side computational
switch with no bearing on what generated the data, and cannot produce a z_true-vs-kernel mismatch.
No `--theta_b`/`--theta_s` flags exist yet anywhere in `arguments.py` (grepped, zero hits) — the
whole θ instrumentation is still unwritten, consistent with the prereg being pre-launch.
**No existing flag achieves the control. This is a factual, not a judgement, finding.**

### 1.5 Verdict

**NEEDS-CODE.** A real S0-R requires (i) a small new realization step that writes a
rescaled-quoted-`z_error` estimator-facing CSV (touches `galaxy_catalogue/observed_realization.py`,
which the repo's own docstring already flags `[PHYSICS] RATIFIED` — a new sibling function there,
not an edit to the ratified `realize_observed_catalogue`, is the lower-risk path) and (ii) new
driver plumbing in `correspondence_1d.py` to carry two decoupled catalogue handlers through one
mirror-seed run instead of one. Until both exist, **PA-HIER-3's own stated consequence stands: no
LEVER-DEAD-AT-N verdict may bank, and D7's early exit is unarmed.** The cheaper fallback the prereg
itself names — disarm D7's early exit and re-scope Stage 0 to S0-A + S0-C only — requires no code
at all and is available today.

---

## 2. PRIOR (PA-HIER-6)

### 2.1 Candidate priors and their consequence for `k`

| # | prior | support / measure | effect on `k` (`σ_h(θ-marginalized)/σ_h(θ=truth)`) and the verdict families |
|---|---|---|---|
| **A** | discrete uniform on the registered grid nodes | exactly the 3×3 or 5×5 grid, equal node weight | `k` is set almost entirely by the grid's own half-width, not by data — this is the finding PA-HIER-6 makes: widening `b`'s half-width from 0.04 to PA-HIER-9's measured 0.163–0.392 (a 4–10× change) changes `k` and can flip FAVOURABLE ↔ UNFAVOURABLE TRADE without a single new likelihood evaluation, because the discrete-uniform prior puts real probability mass at the far corners regardless of how the likelihood actually falls off there. |
| **B** | uniform in `b`, uniform in `ln s`, over an explicitly **stated** continuous support, evaluated by proper quadrature (Simpson on the 5-node axis, trapezoid on 3-node) on the **same already-computed grid nodes** | support = the registered grid extent, stated as a claim rather than left implicit; weights are quadrature weights, not equal-count weights | Same nodes, same cost (zero marginal compute) — this only fixes the *edge-weighting* bias of A (equal-count weighting over-counts corner mass relative to a properly normalized continuum quadrature), not the "support = grid extent, so grid choice sets the verdict" problem, which persists because no likelihood exists outside the registered nodes. |
| **C** | a stated weakly-informative continuum prior (e.g. Gaussian in `b`, Gaussian in `ln s`) with a physically motivated scale (e.g. tied to PA-HIER-9's measured catalogue statistic), evaluated by Gauss–Hermite/Gauss–Legendre quadrature at nodes chosen to match the prior's own natural scale | genuinely open-ended support, decoupled from the grid | Only trustworthy if the quadrature nodes are dense enough near the prior's mass — which in general requires MORE likelihood evaluations than the registered grid provides (a costing consequence: Stage F is already 807–3537 CPU-h; C is not free). If forced onto the *existing* 5 nodes as quadrature abscissas, the quadrature is under-resolved for anything wider than the grid and the option collapses back to B. |

### 2.2 What is forced vs. what the author must choose

**Forced (not a judgement call):** any marginal is a weighted sum over the finite set of nodes
actually computed — there is no way to retroactively query a continuum without new likelihood
evaluations. Given the registered grid, options A and B are the only zero-compute choices; C is
free only in the degenerate case where it reduces to B.

**Judgement (author's call, tag [RULE] or [DO] as appropriate):**
1. Node weighting — discrete-uniform-count (A) vs. quadrature-weighted (B). Low stakes at only
   3–5 nodes per axis, but the choice must be pinned before any `k` is reported, since it changes
   the number by a few percent even at fixed support.
2. Whether the prior's *support* is asserted to equal exactly the registered grid half-width, or
   is a wider physical belief that has merely been **truncated** to what could be afforded — these
   read as the same number but carry different epistemic claims (the first says "θ outside this
   range has zero prior probability"; the second says "θ outside this range is unmeasured, not
   impossible", which should push any REPORTED-ONLY caveat, not a hard CALIBRATED verdict).
3. Whether `s`'s natural parameterization is uniform-in-`s` or uniform-in-`ln s` — the document
   already treats `ln s` as natural everywhere else (log-uniform grid spacing, `score_lns` per
   PA-HIER-4, the `B0-M`/`B0-P` bands), so choosing anything other than uniform-in-`ln s` here
   would be an internal inconsistency; this is *nearly* forced by the document's own prior
   commitments but is technically still a choice the author should ratify explicitly.
4. The coupled **h prior/support** (PA-HIER-6 item (ii), cross-referenced to PA-HIER-14): PA-HIER-14
   found the two cited functions disagree on their h-grid (`H_GRID_41` vs `H_GRID_FULL`,
   `correspondence_1d.py:3788-3789`) and recommends pinning to `H_GRID_41` uniformly. That
   correction, if adopted, resolves this coupling by fixing the h-side of the marginal
   independently of θ's own prior choice.

### 2.3 Recommendation

**Recommend option B** (stated continuum prior — uniform in `b`, uniform in `ln s` — realized by
quadrature weights on the already-computed grid, support pinned explicitly to the registered
half-widths). Rationale: it discharges PA-HIER-6 as a fact-finding matter (a prior now exists, in
writing, with a stated measure) at exactly zero marginal compute; it is the natural reading of the
document's own already-stated log-spacing rationale (§2.3); and it removes the minor
equal-count/quadrature discretization bias of option A for free.

**Two independent sensitivity legs, both free on banked cubes (mark BOTH, not one — they answer
different questions):**
- **Support-width sensitivity** (PA-HIER-6's own registered leg, §2.3 3rd bullet): recompute `k`,
  `t`, the §4.4 rank on the 3-node Stage-P **sub-grid**; a verdict that flips under a *narrower*
  support is REPORTED-ONLY.
- **Weighting-scheme sensitivity** (new, complementary): recompute `k`/`t` under option A
  (equal-count) vs. option B (quadrature) on the **same, full** node set; a verdict that flips
  under the *weighting* alone (support held fixed) is a numerically fragile result and should be
  reported alongside, not silently dropped.

If PA-HIER-9's b-anchor correction is adopted separately (re-measuring `b_max` from the catalogue
rather than the stale `pp_coverage.py` config default), that changes B's stated support and must
be re-run through both sensitivity legs before any width-inflation verdict is read as final —
PA-HIER-6 flags this coupling explicitly and it is real.

---

## 3. IDENTIFIABILITY STATISTIC (PA-HIER-7)

### 3.1 Which of fixed-h / profiled / marginalized matches the registered anchors

The registered anchors, `χ²₂(0.95)/2 = 2.9957` and `χ²₂(0.6827)/2 = 1.1479` (both re-verified:
`-2·ln(1-p)/2` for 2 d.o.f.), are **Wilks profile-likelihood-ratio anchors**. Wilks' theorem gives
an asymptotic `χ²_k` distribution for `-2ΔlnL` between the MLE and a point under a likelihood-ratio
test of `k` restricted parameters, **with any number of additional nuisance parameters profiled
out** — the nuisance count does not enter the degrees of freedom. Here k = 2 (b, s are the tested
parameters) and h is the nuisance parameter. **This is exactly the PROFILED construction**:
`Δ ln L = max_h Σ_i ln L_i(h, truth-θ) − max_h Σ_i ln L_i(h, corner)`, matching PA-HIER-7's own
registered correction. Recommend: **pin PROFILED as the band-bearing statistic.**

- **FIXED-h** (`h ≡ 0.73` on both sides) has no valid χ²₂ correspondence when a genuine h–θ
  degeneracy exists, because it never lets the "best available" h absorb any of the corner's
  displacement — it measures a strictly larger, and generically inconsistent, quantity.
- **MARGINALIZED** (`∫ dh L(h,θ)·π(h) dh`) has no general Wilks-type asymptotic guarantee at all —
  Bayesian evidence ratios and profile likelihood ratios coincide only under restrictive regularity
  and flat-prior conditions that a 41-node, possibly boundary-railing h-posterior (the repo's own
  documented H0-railing pathology is exactly this kind of non-regular behavior) cannot be assumed
  to satisfy. Applying the χ²₂ bands to a marginalized ΔlnL is not merely non-standard, it is
  **unfalsifiable as posed** — there is no citable anchor for it.

### 3.2 How the three differ along a ridge (the prereg's own anticipated structure)

Given a strong h–b ridge (§4.2's own expected secondary read):

- **FIXED-h** at `h = 0.73`: moving from truth-θ to the corner-θ, with h pinned, crosses the
  ridge's full **transverse** curvature — the steep direction — because h is not allowed to
  re-adjust along the ridge to partially compensate the b-shift. This *overstates* identifiability:
  a fixed-h ΔlnL can clear the 3.00-nat bar while the operationally relevant (profiled) number sits
  well inside UNIDENTIFIABLE.
- **PROFILED**: for each θ node, first re-maximizes over h — i.e. "climbs back up the ridge" to the
  best-fit h for that θ — before comparing. Only the **ridge-transverse residual** (the curvature
  that h-adjustment cannot absorb) survives into ΔlnL. This is the number that answers the thread's
  actual question: once h is jointly inferred (as production does), does the data still pin down
  θ, or does the ridge let θ roam freely while h silently compensates?
- **MARGINALIZED**: integrating over h (rather than taking its single best value) further dilutes
  the truth-vs-corner contrast whenever the h-posterior along the ridge has non-trivial width —
  generically producing an even smaller number than PROFILED, and one that is additionally
  prior-dependent (coupling directly back into §2's open prior choice). Because it lacks an
  asymptotic anchor, it cannot be band-bearing regardless of its numeric value.

### 3.3 What to report alongside

- The fixed-h and marginalized variants, REPORTED-ONLY, so the ridge's own size is visible even
  though only PROFILED is band-bearing.
- The best-fit `h(θ)` value at every grid node used in the profile (this is the ridge itself,
  directly plottable, and is free — it already exists inside the profiling computation).
- `ρ(h, b)` and `ρ(h, ln s)`, the pooled degeneracy correlations §4.2 already registers as a
  secondary read — report alongside the ΔlnL number, not as a separate pass.
- PA-HIER-7's own registered precondition: `lnL(truth-θ) ≥ lnL(θ)` for every other Stage-P node
  (i.e. truth-θ really is (locally) the maximum-likelihood point on the grid); if it fails, the
  "truth ≈ MLE" premise the Wilks anchor requires does not hold on a 3×3/5×5 grid and the
  IDENTIFIABLE/UNIDENTIFIABLE read must be downgraded to REPORTED-ONLY for that arm, exactly as
  PA-HIER-7 already registers.

---

## 4. WIRING TEST (PA-HIER-16)

### 4.1 Why GATE T-ID cannot detect a missing hook (re-confirmed)

§3.1's literal early-return at `(b, s) == (0.0, 1.0)` is the right call for byte-identity (IEEE-754
reordering: `host_z_error * 1.0` is exact, `sqrt(x**2*s**2 + pv**2)` is not guaranteed to round
identically to `sqrt((x*s)**2 + pv**2)`) — but a path with **no hook at all** produces the identical
output at `θ = (0,1)`, since the early-return and "there was never anything to return early from"
are indistinguishable at that one point. T-ID certifies the default only; it says nothing about
wiring.

### 4.2 Why §3.4's ENG, as written, is blind to a single missing site

ENG today runs at the Stage-P corner with **every** in-scope site's hook requested simultaneously
and checks an aggregate "≥10% of events moved". If exactly one site is unhooked while the others
are correctly wired, the live sites alone already clear 10% — **the aggregate criterion passes
trivially even when one specific site is silently dead.** This is the structural gap: ENG as
registered can only distinguish "everything moved" from "nothing moved", never "site X specifically
didn't move".

### 4.3 Concrete path-isolated toggle-matrix design

For each in-scope **estimator-side** dispatch site — 2.1 (`bayesian_statistics.py:6223-6224`
scalar), 2.2 (`:6878-6879`/`:6899-6901` batch — production's actual dispatch path), 2.3
(`:1619-1720`/`:2657-2882`, the global selection denominator) — run an **OAT (one-at-a-time)**
matrix, not a full 2³ factorial (sufficient to prove/disprove each site's individual wiring; a
factorial would additionally catch site-*interaction* bugs but at 2³× the compute for a case this
gate is not chartered to catch):

| run | θ requested | hook feature-flag state |
|---|---|---|
| baseline | (0, 1) | all sites disabled (= today's T-ID) |
| site-2.1-only | (0.02, 1) or (0, √2) | 2.1 active, 2.2/2.3 forced to their θ=(0,1) evaluation |
| site-2.2-only | same θ | 2.2 active, 2.1/2.3 forced |
| site-2.3-only | same θ | 2.3 active, 2.1/2.2 forced |
| all-sites | same θ | current registered design (all active) |

Note site 2.4 (`correspondence_1d.host_z_error_eff`) is **excluded** from this matrix by
construction — PA-HIER-2's GATE GEN-FROZEN forbids any θ hook there at all (it is generator-side);
including it in a θ-wiring test would itself be the PA-HIER-2 violation.

**Measurement, per run, decomposed per event into numerator-log-term and
denominator-log-term separately** (a light instrumentation addition if not already exposed — the
existing per-event diagnostics CSV, per CLAUDE.md's "every `--evaluate` run emits a per-event
both-channel diagnostics CSV", is the natural home for this):

- **Sites 2.1/2.2 (numerator, per-host):** expected signature is a **heterogeneous** shift —
  different hosts have different `(z, z_error)`, so a live `s`/`b` hook must move each event's
  numerator log-term by an amount that depends on that event's own host parameters. **Present:**
  ≥10% of events move ≥1e-6 relative AND the per-event magnitude correlates with each host's own
  `z_error`/`z` (spot-check 2–3 hosts against a hand-computed `ln N(z; z_g + b(1+z_g), s·σ_z)`
  closed form, to machine precision). **Missing:** zero movement in the numerator-log-term
  specifically, in the run where ONLY that site's flag was set — decisive, regardless of what the
  all-sites run shows.
- **Site 2.3 (global denominator, per-h scalar):** expected signature is the **opposite** —
  a single shared multiplicative/additive shift applied identically to every event's
  denominator-log-term (this IS the correct signature here, not a red flag; site 2.3 is a scalar
  by physical construction). **Present:** (i) the shift is bit-identical in magnitude across every
  event (confirms it truly is shared/scalar, not silently varying); (ii) the shift is absent
  (bit-identical to baseline) in the site-2.1-only and site-2.2-only runs (confirms isolation
  actually held); (iii) the shift matches an **independent recomputation** — evaluate
  `_smeared_global_pdet_expectation`'s formula directly at the same `(h, θ)` outside the full
  pipeline and require exact agreement with the pipeline-produced shift. **Missing:** zero
  denominator-log-term movement in the site-2.3-only run — decisive even though this is exactly
  the case where "N% of events moved" would otherwise read as vacuously true (100% "move" in
  aggregate ln L terms if 2.1/2.2 are also live, or 0% if everything including 2.1/2.2 is
  disabled — the per-term decomposition is what breaks the ambiguity, not the aggregate count).
- **Cross-check the isolation claim itself:** in the site-k-only run, every **other** in-scope
  site's own decomposed term must be bit-identical to baseline (T-ID-style), not merely
  "small" — this verifies "all others forced to θ=(0,1)" was actually enforced by the driver,
  rather than assumed.

**Complementary, not a substitute:** PA-HIER-16's own registered hook-inventory assertion (driver
asserts each site's θ-aware code object was imported and its counter incremented at least once per
task, stamped into the task JSON) is a cheap static check that the code path was *entered*, but
does not prove the hook's *arithmetic* took effect (a hook can be entered and still compute-then-
discard its result). The toggle-matrix numeric test above is the decisive evidence; the inventory
assertion is a free, complementary sanity check on top of it.

---

## 5. Per-blocker disposition

| blocker | disposition | one-line summary |
|---|---|---|
| **PA-HIER-3** (control) | **NEEDS-CODE** | Re-verified from source: `z_error` is a pure passthrough and one handler feeds both generator and estimator at every `sigma_scale` — no existing flag decouples them. A genuine control needs a new estimator-facing catalogue (rescaled quoted `z_error`) plus new driver plumbing to carry two catalogue handlers through one mirror-seed run. Until built, no LEVER-DEAD-AT-N verdict may bank; the free fallback (disarm D7's early exit, re-scope Stage 0 to S0-A+S0-C) needs no code. |
| **PA-HIER-6** (prior) | **OPEN-FOR-AUTHOR**, with a recommendation | Fact-side is settled: no prior is registered anywhere, and the grid's own extent silently sets `k`/the width-ratio verdicts under an implicit discrete-uniform-on-grid reading. Recommend option B (stated continuum prior, quadrature-weighted on the existing nodes, zero marginal compute) with both the support-width and the weighting-scheme sensitivity legs run before any FAVOURABLE/UNFAVOURABLE-TRADE or CALIBRATED verdict is read as final. The one-line author choice: adopt B, or state a different prior explicitly — either closes the blocker; silence does not. |
| **PA-HIER-7** (identifiability statistic) | **RESOLVED** (a matter of fact) | The registered χ²₂ anchors are profile-likelihood anchors; only the PROFILED statistic (`max_h` on both sides) matches them, matching PA-HIER-7's own registered correction. FIXED-h overstates identifiability along an h–b ridge; MARGINALIZED has no valid asymptotic anchor at all and must stay REPORTED-ONLY regardless of its value. Pin PROFILED as band-bearing; report the other two plus the ridge shape (`h(θ)` at each node, `ρ(h,b)`, `ρ(h,ln s)`) alongside. |
| **PA-HIER-16** (wiring test) | **RESOLVED** (a concrete design exists) | GATE T-ID certifies the default only; the registered aggregate-ENG form is blind to exactly one missing site whenever the others are live. The path-isolated OAT toggle matrix above (one site's hook active at a time, decomposed per-event numerator/denominator terms, plus an independent-recomputation cross-check for the scalar site 2.3) detects a missing hook at any single in-scope site regardless of the others' state, and correctly treats site 2.3's "100% of events move by the same amount" as its expected — not vacuous — positive signature by checking the shift's uniformity and independent-recomputation match rather than its raw event-count coverage. |

---

## 6. Cross-cutting note (not a fifth blocker, flagged for completeness)

Every design above for the control (§1) explicitly threads decoupled generator-side and
estimator-side catalogue objects through the harness. This interacts directly with PA-HIER-2's
GATE GEN-FROZEN (no θ hook may reach `correspondence_1d.host_z_error_eff` /
`kernel_smeared_survival` / `_draw_kernel_survival_redshifts`, and the venue's realized per-seed
event table must be byte-identical across every θ node at fixed seed). The control's new
estimator-facing catalogue must be built so that it changes **only** what `bs.evaluate`'s
`galaxy_catalog` argument reads, never `draw_realization`'s inputs — i.e. it must satisfy
GATE GEN-FROZEN by construction, not by accident. This is a design constraint on the control's
implementation, not a new blocker.
