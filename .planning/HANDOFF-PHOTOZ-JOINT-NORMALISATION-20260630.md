# HANDOFF — Deep joint-normalization fix for photo-z dark-siren H₀

**Read this, then DISCUSS the solution hints (§4–5) before kicking off deep research.**
This is a *framing* document, not a research plan. It exists because the last two
investigations succeeded by discussing the problem first; do the same here. The fix is a
real methodological reconstruction, and the wrong framing wastes a lot of compute.

Starting point: tag `photoz-railing-v1` (commit `ee98f71`). Full record:
`scripts/bridge_closure/BRIDGE-FINDINGS.md`, `.planning/derivation-photoz-incatalog/`
(esp. `NORMALISATION-FIX.md`), `.planning/RELEASE-STATUS-PHOTOZ-RAILING-20260630.md`,
memory `h0-railing-rootcause-photoz`.

---

## ▶ AGREED STARTING POINT (decided 2026-06-30 — DO THIS FIRST, in a fresh session)
**The first task is a COMPARISON: map our partition-norm pipeline against the existing
*unbiased* photometric dark-siren methods in the literature, and produce a side-by-side
gap analysis.** Rationale: a working unbiased photometric method already exists (Echoes
2509.18243 / 2502.17747; GWCosmo, Gray 2023; Chen/Fishbach 2212.08694; Gray 2020
1908.06050), so the correct form is known — don't re-invent it; reverse-engineer it and
find the single piece our pipeline is missing. This comparison is the *initial analysis*;
the exploratory directions (§4 hints) follow FROM it, they don't precede it.

**Scope of the comparison (for each literature method, then OURS):**
- the per-galaxy redshift term — is `p(z|z_g)` a likelihood or a prior; how does the
  photo-z enter (convolution? marginalisation?);
- the selection function β(H₀) WITH photo-z — and crucially **does the catalogue density
  appear identically in the numerator and the denominator?**;
- the normalisation LEVEL — per-event vs population/ensemble/hierarchical;
- the injection / simulation of the photo-z — is it self-consistent (object-for-object)?;
- the VALIDATED REGIME — σ_z/z, completeness, z-range, p_det behaviour. **Is the
  GLADE-at-z≈0.05 regime (σ_z/z≈0.7, p_det≈1) even inside their demonstrated range, or do
  they implicitly require a varying selection / higher z / smaller σ_z/z?**

**Deliverable:** a `COMPARISON.md` (method-by-method extraction) + a `GAP-ANALYSIS.md`
(our-pipeline-vs-theirs table, the specific divergence(s) that cause our railing, ranked)
under `.planning/derivation-photoz-incatalog/`. **STOP there and present the gap analysis
for discussion before any derivation or code change.** Suggested shape: a research workflow
fanning out one reader per paper + one agent mapping our partition-norm structure from the
code, then a synthesis. (Ultracode workflow is appropriate.)

Do this on branch `physics/photoz-joint-normalisation` (off tag `photoz-railing-v1`).

---

## 1. The problem in one paragraph
The in-catalogue dark-siren H₀ likelihood **rails** (monotonic, non-peaked posterior to
the grid edge) when host redshifts are **photometric** — GLADE flag-1 hosts have
σ_z ≈ 0.035, which at z≈0.05 is **~10–18× the GW redshift precision** (σ_z^GW ≈ 0.037·z
≈ 0.002). Once σ_z ≫ σ_z^GW the GW distance no longer localizes the host, so the inference
stops being *data-dominated* and becomes *prior/selection-dominated*: the rising galaxy
redshift prior n(z) ∝ dV_c/dz drags the posterior to the edge. **It is unbiased in the
data-dominated (spectroscopic) limit** — so this is a *normalization* problem exposed by
large σ_z, not a fundamental "large errors = bias" law (this was confirmed; see §3).

## 2. The proven obstruction (why the easy fixes can't work)
The numerator collapses (sharp GW factor) to `A(h)·p_eff(z*(h))`, where z*(h) =
`dist_to_redshift(d_meas, h)` **increases with h**, and `p_eff` is the numerator's
effective redshift prior. The single **global scalar** selection denominator D(h) does NOT
track the *local* gradient of `p_eff` at z*(h). The deep reason:

> **`p_det ≈ 1` across the entire in-catalogue redshift range** — detected EMRIs sit at
> z≈0.05, far inside the GW horizon, so the selection function is **flat where the galaxies
> are**. A selection-based normalization can only cancel a density gradient *where the
> selection varies*. Here it doesn't, so D(h) (which moves only via the distant horizon
> edge) is blind to the local catalogue-density gradient.

Consequence (measured in the clean `rung_I` closure, truth 0.73):
- standard `p_eff = n_smooth` (doubly-smeared dV_c) → rails **down** to 0.60;
- cleaned `p_eff = p_bg ∝ dV_c/(1+z)` (still rising!) → rails **up** to 0.87;
- **truth sits strictly between the two rails.** Cleaning the numerator only flips which
  way it tips. *No per-event, numerator-only normalization lands on the truth.* (Proven.)

## 3. What is RULED OUT — do not retread
- **Every non-redshift hypothesis** (each recovers when added alone in the bridge):
  Malmquist (selection is on the *true* SNR), σ²/distance scatter at production N, the
  catalogue n(z) density *shape*, sky / 3-D MVN / Fisher correlations, candidate selection
  / ball-tree / coordinate frame (returns the true host 99% of the time), completion B_num,
  survival p_det, pixelated f_k, candidate radius, **D(h) magnitude** (a confirmed red
  herring — needs an absurd power ~1000 to matter).
- **Spec-z-only inference filter** → still rails (events were injected from photo-z hosts;
  filtering one side breaks sim↔inference consistency).
- **All numerator-only normalizations** (per-galaxy posterior `N·p_bg/Z_g`; global de-count
  `p_bg/(S·p_bg)`) → pass the σ_z→0 gate but do **not** de-rail (proven, §2).
- **Local "consistent denominator"** → rails at *all* σ_z (breaks Option-A global
  scale-freedom — disqualified).
- **THE σ_z→0 GATE (hard constraint):** any candidate MUST reduce to the standard Option-A
  *global* form as σ_z→0 (where the posterior is already peaked/unbiased). A candidate that
  changes the small-σ_z behaviour is disqualified immediately, before checking high σ_z.

## 4. Solution hints (the candidate directions — to discuss, not yet chosen)
The fix is **outside** the numerator-only space. Four interlocking hints:

1. **Joint num+denom GLOBAL same-kernel ratio.** The genuine Gray/Hitchhiker form puts the
   *identical* population density `p_cat(z)` in BOTH the numerator and the selection
   denominator, over the *full* redshift range: `∫ p_GW·p_cat / ∫ p_det·p_cat`. This is a
   **denominator change** (so it leaves the "freeze the global denom" box we searched). ⚠
   Caveat to settle: with `p_det ≈ 1` this likely *degenerates* per-event
   (`∫ p_det·p_cat ≈ ∫ p_cat = W`), so the ratio alone may not de-rail. Is it real?

2. **Ensemble coherence is probably the actual lever.** If per-event normalization is
   provably blind to the local gradient (§2), the de-railing must come from the **collective**
   constraint: many events' z*ᵢ(h) loci shift together with H₀, while their per-event
   density-gradient "noise" is what should average/normalize out. The open question: *what
   object* supplies this — it is NOT a per-event normalization. (Hierarchical population
   likelihood? A shared H₀-dependent normalization across the event set?)

3. **Photo-z-consistent re-injection (sim side).** Today the events are injected at the
   host's *exact* catalogue z, while the inference convolves the *reported* z_g — a residual
   sim↔inference inconsistency (object-for-object mismatch). The consistent setup: the
   galaxy's true z is drawn, the catalogue reports z_g = z_true + N(0,σ_z), the EMRI is at
   the true z, and the inference convolves σ_z around z_g. **Likely a precondition** for any
   normalization to be unbiased at fixed catalogue. (NB: this implies the production CRB run
   may need re-simulation, not just an eval-side change.)

4. **Don't double-count the comoving volume (Hitchhiker "Inconsistency 1").** The host-z
   prior should be the *population* p_pop(z), with the catalogue providing the host
   *likelihood* (not the prior). Conflating the catalogue density n(z) with the prior puts
   dV_c in twice. The correct construction makes n(z) a sampling artifact that cancels,
   leaving p_pop — and must still reduce to Option-A as σ_z→0.

## 5. The single most important pre-research question
**The literature has a working unbiased photometric method** (Echoes, arXiv:2509.18243 /
2502.17747; GWCosmo, Gray 2023; Chen/Fishbach "Hitchhiker's Guide" 2212.08694). So a correct
form *exists*. **Before deriving anything, reverse-engineer EXACTLY what they do:**
- What is their per-galaxy redshift term (likelihood vs prior; what p(z|z_g))?
- What is their β(H₀) / selection normalization with photo-z — and crucially, *does the
  catalogue density appear identically in numerator and denominator*?
- Do they normalize per-event or at the population/ensemble level?
- How do they inject / simulate the photo-z (the consistency question of hint 3)?
- In what regime do they demonstrate unbiasedness (σ_z/z, completeness, z-range)? Is the
  GLADE-at-z≈0.05 regime (σ_z/z ≈ 0.7, p_det≈1) inside or outside their validated range?

Map their answer onto our partition-norm structure, identify the single missing piece, and
*then* derive. This is the highest-leverage step — they solved it; we should not re-invent.

## 6. Validation harness + pass criteria (already built)
`scripts/bridge_closure/rung_I_prior_domination.py` — the **clean, self-consistent, no-sky
closure** is the arbiter (no sky/MVN/window artifacts; reproduces the baselines exactly).
Add each candidate behind a flag in `run_closure_photoz(...)`.
- **GATE (necessary):** σ_z = 0.002 → MAP ~0.73, PEAKED (must match standard's 0.744).
- **DE-RAIL (sufficient):** σ_z = 0.035 → MAP ~0.73, **interior peak** (beat standard's 0.600).
- Baselines to beat: standard 0.744 / 0.600; all numerator-only fixes 0.748 / 0.870.
- Prototype scaffold with all prior candidates side by side: `scripts/bridge_closure/_rungI_verify_B.py`.
- ⚠ The full sky `conv` mode (`_bridge_sky.py`) has window/grid artifacts — **use `rung_I`
  (no-sky) as the quantitative arbiter**, not the sky rungs.

## 7. Related open item — σ_M (mass error)
The with-BH-mass (4-D) channel uses the catalogue/Fisher **mass** uncertainty σ_M, also
large in the current setup. The same prior/normalization-domination could affect it. Not yet
investigated — keep it in scope (a correct fix should ideally be channel-agnostic).

## 8. Suggested sequencing (for the discussion to confirm/adjust)
1. **Literature reverse-engineering** (§5) — the working photo-z normalization, exactly.
2. **Decide the lever:** joint same-kernel ratio vs ensemble/hierarchical normalization vs
   re-injection — likely a combination. Resolve the `p_det≈1` degeneracy question first.
3. **Derive** the minimal change to the partition-norm structure, with the σ_z→0 reduction
   proof (the gate) up front.
4. **Prototype in `rung_I`**, gate + de-rail test, before any `bayesian_statistics.py` edit.
5. If it passes: `[PHYSICS]` production change + (likely) photo-z-consistent re-simulation,
   then a consolidated cluster run.

## 9. Branch
Create `physics/photoz-joint-normalisation` off tag `photoz-railing-v1`. Keep the σ_z/σ_M
realization study on its own branch (`study/sigma-z-m-realizations`) — independent path.
