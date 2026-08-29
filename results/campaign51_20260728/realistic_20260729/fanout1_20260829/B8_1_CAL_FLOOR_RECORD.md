# B8.1 [CAL] — the F5 information floor at the PRODUCTION venue (1D + 2D)

`launched under rows #222/#223 — charter node B8.1`

**Class:** measurement (Fisher/Cramér-Rao floor computation on real production data), not
a hypothesis test. Builder = this agent; per standing rule 2 (verifier independence) the
numbers below must be **reproduced by a different agent re-running the instrument**
(`b8_information_floor.py`) before they are cited as adjudicated — this record is the
builder's smoke-tested output, not an independently verified one.

**Bottom line (one sentence):** the idealized, single-known-host, no-impostor Fisher
floor at the actual N=1588 production event set is **σ_h,floor ≈ 0.0017 (0.24% of h)**
for both channels at realistic host errors — the with-BH-mass channel adds essentially
**zero** rescue at any literature-realistic σ_M (confirming F5) — and the **measured**
production posteriors sit **≈11× wider** (2D) and the 2D **centre misses truth by ≈38
floor-σ**, so the campaign is not simply "information-starved" in the benign sense; it is
leaving ~99% of the theoretically available single-host Fisher information on the table
(§3), which is the number this node was asked to produce.

**Exoneration check (standing rule 5, both layers grepped):** `EXONERATION_REGISTER_20260827.md`
§13 **[INFO-STARVATION] is OVERTURNED** — "information starvation" is banned as *the
explanation for the H0 rail* ("a property of prior-INCONSISTENT estimators, not of the
data … Do NOT resurrect it as an explanation"). This node's finding does **not** resurrect
it: the floor computed here shows the *opposite* of "the data has too little information" —
the theoretical single-host information is tight (§2.1), and the measured/floor gap (§3) is
presented throughout as an **estimator-consistency / confusion signature**, matching, not
contradicting, the overturned verdict's own resolution ("consistency is the actual cure").
The charter's own root-goal quote (runbook 37 §0, verbatim) uses "starved" only as the
**ceiling** past which no further estimator fix can help — this record supplies that
ceiling number; it does not claim starvation as a present-tense cause of the rail.

---

## 0. Data and method (read first)

- **Event set (N=1588):** `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`
  — the CRB/event set **shared, md5-verified, by both HEAD-readout venues** `iiib` and
  `joint_r1` (`MEASUREMENT_HEAD_READOUT_20260827.md` §1.1: "CRB / event set … same file
  (symlink)"; venues differ only in which galaxy catalogue is read at evaluate-time, not
  in the injected GW event set). Filtered to production's own cuts — `SNR>=20`
  (`constants.SNR_THRESHOLD`) and fractional distance error `<10%`
  (`bayesian_statistics.py:386` `FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD`) —
  1590 rows → **1588 events**, exactly matching the "Events scored: 1588" both venues
  report. {value: 1588, source: `b8_information_floor.json:meta.n_events`, date:
  2026-08-29}
- **Per-event inputs used, all ACTUAL (none simulated/assumed):** GW-measured luminosity
  distance `d_meas` and its CRB 1σ `sigma_dL = sqrt(delta_luminosity_distance_delta_luminosity_distance)`
  (median fractional error **3.73%**, p10–p90 **1.7%–5.1%** — `b8_information_floor.json:meta.frac_dL_err_median`);
  GW-measured detector-frame mass `M_z` and its CRB 1σ `sigma_Mz` (median fractional error
  **8.8×10⁻⁸** — the GW mass channel is essentially exact; the entire 2D error budget is the
  *host* photo-mass term, not the GW measurement). Host redshift `z_i` is recovered from
  `d_meas` by inverting the production distance law at `h_true=0.73`
  (`physical_relations.dist_vectorized`, production cosmology `OMEGA_M=0.2726`,
  `OMEGA_DE=0.7274` — **not** the F5 engine's own internal `_bridge_lib` toy cosmology
  0.25/0.75, which is deliberately not used here). Median z = **0.490**, p10–p90
  **0.237–0.736**, max **1.02** — this population reaches much deeper than the original
  June F5 sweep's toy self-consistent population (median z≈0.15, horizon 1.2 Gpc); see
  caveat §5.6.
- **Realistic (stipulated, not measured) inputs:** host photo-z error `sigma_z` = GLADE+
  flag-1 photometric median **0.035**, or flag-3 spectroscopic median **0.0017**
  (`sigma_z_sigma_M_forecast.py:72-73`); host photo-mass fractional error `sigma_M` from
  the Reines & Volonteri (2015) intrinsic-scatter table in
  `docs/MASS_RELATION_ASSESSMENT.md` §2 — **0.19** (the code's current fit-only estimate,
  a known 3–7× under-estimate), **0.60** (0.24 dex intrinsic floor), **1.66** (0.50 dex
  measurement), **1.99** (0.55 dex total predictive, the realistic number), plus **0.02**
  (the F5-quoted "useful" threshold, kept only as an informational anchor — not a
  literature-supported achievable value).
- **Instrument:** `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b8_information_floor.py`.
  Outputs: `b8_information_floor.json` (this record's numbers, `{value, source: file:line
  in the script, date: 2026-08-29}` for every quoted figure below).

---

## 1. The Fisher/information argument (item 1)

The production distance law factorizes exactly as `d_L(z,h) = D(z)/h` with `D(z)=dist(z,1)`
h-independent (`physical_relations.dist:132-` and the same optimisation the F5 engine
already uses, `sigma_z_sigma_M_forecast.py:162-169`). For one host known with certainty
(no impostor competition — the idealization this node is chartered to define), the
observables are:

- the GW distance `d_meas ~ N(D(z_true)/h, sigma_dL²)`,
- the host's catalogue redshift `z_cat ~ N(z_true, sigma_z²)`,
- (2D only) the GW detector-frame mass `M_z_meas ~ N(M_g(1+z_true), sigma_Mz² + (sigma_M M_g(1+z_true))²)`,
  mirroring `bayesian_statistics.py`'s `with_bh_mass` numerator term and the F5 engine's
  `_accumulate()` (`sigma_z_sigma_M_forecast.py:329-337`) exactly.

The joint Fisher matrix in `(h, z)` at the true point, using `∂d_L/∂h = -d_L/h` and
`∂d_L/∂z = D'(z)/h` (`D'≡dD/dz`), is

```
F_hh = (d_L/h)² / σ_dL²          F_hz = -(d_L/h)(D'/h) / σ_dL²          F_zz = (D'/h)² / σ_dL² + 1/σ_z,eff²
```

with `1/σ_z,eff² = 1/σ_z² + 1/(σ_M(1+z))²` combining the photo-z and (2D) mass-anchor
redshift constraints by inverse variance (the mass anchor is h-independent, so it only
ever tightens `σ_z,eff`, never `F_hh` directly — this is the F5 mechanism, restated as a
Fisher term). Profiling out the nuisance `z` (Schur complement, `I_h = F_hh - F_hz²/F_zz`)
gives the single-host information

```
I_i = d_L,i² / [ h² σ_dL,i² + D'(z_i)² σ_z,eff,i² ]         σ_h,floor(N) = ( Σ_i I_i )^(-1/2)
```

This is implemented as `closed_form_fisher()` in the instrument (an earlier draft
multiplied *both* denominator terms by `h²` — an algebra slip caught before this was
registered; only the `σ_dL²` term carries the Jacobian factor, see the docstring in the
script for the derivation check). Dimensional check: `[d_L²]=Gpc²`, `[h²σ_dL²]=Gpc²`,
`[D'²σ_z²]=Gpc²` — consistent, and `I_i` has units `h⁻²` as required for a Fisher info on
`h`. Limiting cases: `σ_M→0` recovers the bare photo-z-only 1D channel exactly (mass term
drops out of `σ_z,eff`); `σ_dL→∞` (undetected) sends `I_i→0`; `σ_z→0` (perfect host z)
sends `I_i → (d_L/h)²/σ_dL²`, the pure-GW-distance information (the ceiling this floor can
never exceed for a given event, since the mass anchor can only ever *substitute* for the
photo-z, not beat a perfect one).

---

## 2. Evaluation at N=1588 (item 2)

### 2.1 Route B — closed-form (route of record)

| channel | σ_z | σ_M | σ_h,floor | floor/h | n_eff / 1588 | top-10 share |
|---|---|---|---|---|---|---|
| 1D | 0.035 (photo) | — | **0.001747** | 0.239% | 1266 | 1.4% |
| 1D | 0.0017 (spec) | — | 0.000560 | 0.077% | 898 | 3.1% |
| 2D | 0.035 | 0.02 (F5 threshold, informational) | 0.001295 | 0.177% | 1397 | 1.1% |
| 2D | 0.035 | 0.19 (code fit-only) | 0.001738 | 0.238% | — | — |
| 2D | 0.035 | 0.60 (0.24 dex intrinsic) | 0.001746 | 0.239% | 1266 | 1.4% |
| 2D | 0.035 | 1.66 (0.50 dex) | 0.001747 | 0.239% | — | — |
| 2D | 0.035 | 1.99 (0.55 dex, realistic) | **0.001747** | 0.239% | 1266 | 1.4% |
| 2D | 0.0017 | any of the above | 0.000560 | 0.077% | 898 | 3.1% |

{values: `b8_information_floor.json:oneD/twoD.*.closed_form`, source:
`b8_information_floor.py` (`closed_form_fisher`, `summarize`), date: 2026-08-29}

**Reading it:** at GLADE photo-z, the mass channel only helps at the informational
`σ_M=2%` anchor (floor tightens 26%, 0.001747→0.001295) — and even that is a **26%**
tightening, not the ~50× the idealized synthetic-population F5 sweep advertised (§5.6
explains why real production z's differ from the June toy). At every literature-realistic
σ_M (0.19–1.99), the 2D floor is **indistinguishable from the 1D floor** — the with-BH-mass
channel adds **no rescue** at production N, confirming (at the actual N, for the first
time, not by N-rescaling a toy sweep) the F5/`MASS_RELATION_ASSESSMENT.md` conclusion.
`n_eff` (Kish effective event count) sits at 80–90% of N with no event contributing more
than a few percent — the floor is **not** an artifact of one or two golden events; it is a
smooth, well-spread sum. Spectroscopic σ_z alone (no mass term at all) already beats every
photometric+mass combination — redshift precision, not mass precision, is what would move
this floor.

### 2.2 Route A — numeric (a documented negative result, not a headline number)

A second, nonlinear/marginalized route was built (`numeric_fisher()`: exact
`d_L(z,h)=D(z)/h`, no local linearization, per-event log-likelihood integrated over a
z-grid, curvature taken by 3-point finite difference in `h`). It **agrees with Route B to
4 significant figures in the spectroscopic regime** (`σ_z=0.0017`: both give
`σ_h,floor=0.000560` exactly) — a genuine cross-validation of the Route-B algebra where
the model is well inside its linear regime. But at GLADE photo-z (`σ_z=0.035`, comparable
to or larger than the low-z tail of the population) it **disagrees by ~5×** and is
**dominated by 10 events out of 1588 (88% of the total, n_eff≈5)** — all at z≈0.02–0.08,
the shallowest end of the catalogue. Diagnosing one such event (idx 889, z=0.0213,
fractional distance error 0.09%) shows why: its marginal log-likelihood is **not peaked**
at h_true within ±0.17 in h — it is nearly monotonic, because the wide photo-z kernel
(σ_z=0.035 ≫ z=0.02) lets the nuisance z "chase" any trial h almost for free (the
classic h–z degeneracy of a dark-siren single host). A 3-point finite difference with
`dh~0.005` on a locally near-flat, monotonic function is numerically unstable — it returns
a spuriously enormous curvature — and the instrument's own `dh`-convergence sweep confirms
this (`σ_h,floor` at `dh=0.002,0.005,0.01,0.02` = `0.000323, 0.000372, 0.000486, 0.001041`
— growing with `dh`, not converging, the signature of a non-quadratic/degenerate local
shape being aliased into a spurious curvature). **Conclusion: Route A is unreliable
precisely in the photometric regime that matters, and Route B (the smooth algebraic
Fisher matrix, immune to this finite-difference failure mode by construction) is the
number of record.** This is itself a useful, reportable finding, not a discarded scratch
result: it is direct, at-production-N evidence that the single-host marginal posterior is
**severely non-Gaussian/degenerate** at low z under GLADE photo-z — see §2.3.

### 2.3 Reconciling with the June F5 engine's own headline (§5.6 detail)

The June `sigma_z_sigma_M_forecast.py` sweep (toy N=400, synthetic population, median
z≈0.15) reported the 1D channel **railed** at GLADE σ_z (`rail_frac→1`, `σ_eff≈26-30%`
of h — `docs/SIGMA_Z_SIGMA_M_FORECAST.md` §2). That is a **finite-sample, fully nonlinear
Monte-Carlo posterior width**, not a Fisher floor — it is exactly the same h–z degeneracy
that broke Route A above, but realized through an actual noisy simulated dataset rather
than an unstable finite difference. Cramér–Rao guarantees `Var(actual estimator) ≥
1/Fisher`; it does not promise the two are close when the likelihood is this degenerate.
The gap between the June engine's ~26–30% railed width and this node's ~0.24% Route-B
floor is not a contradiction — it is a second, independent confirmation (on top of §2.2's
own finding) that **even a single perfectly-identified host's marginal posterior is far
more degenerate than its local curvature at truth suggests**, before any impostor
competition is added at all. This also lines up with `BIAS_HISTORY_LEDGER.md` row #98's
own DS-8 target T1 ("single-host starvation rail 400/400 ×3 truths," CONFIRMED
2026-08-10/11 in a controlled multi-candidate-ball instrument) — a third, independently
built instrument reproducing the same single-host degeneracy class. Row #98 also found
this rail does **not** reproduce in the *multi-candidate* ball venue at any σ_z dose
(DS-6 MIXED, "the in-loop defect is a coverage collapse, not a rail") — i.e. once impostor
competition is added the failure mode changes shape again, which is exactly why §5.1's
single-host caveat matters and why this node's floor cannot be read as a calibration
target on its own.

---

## 3. Comparison with the current HEAD posteriors (item 3)

Measured values quoted verbatim from `head_readout_extraction_20260827.md` (its own
"Results table"; {source: that file, date: 2026-08-27}), venue-averaged (iiib, joint_r1):

| channel | measured ⟨σ_h⟩ | measured ⟨bias⟩ | floor (realistic) | width/floor | \|bias\|/floor | RMSE/floor |
|---|---|---|---|---|---|---|
| 2D | 0.01847 | −0.0668 | 0.001747 | **10.6×** | **38.2×** | **39.7×** |
| 1D | 0.00996 | −0.1190 | 0.001747 | 5.7× (misleading, see below) | **68.1×** | **68.4×** |

- **2D is the meaningful width comparison** (its posterior peaks away from the grid
  boundary, `MAP≈0.663–0.665`, not pinned at the edge). At realistic σ_M its measured
  width is **~11× the floor** — equivalently, production captures only **~1/112 ≈ 0.9%**
  of the idealized single-host Fisher information (width ratio squared). Its *bias* is
  worse still: the 2D peak misses truth by **38 floor-σ**, an order of magnitude beyond
  what "not enough information" can explain on its own — this is the signature already
  named elsewhere in the ledger (impostor drag, window geometry, tilt) rather than a new
  finding of this node, but it is now a **quantified** one against a real floor for the
  first time.
- **1D's `σ_h=0.0085–0.0115` is not a fair width comparison** — it is the standard
  deviation of a distribution **pinned at the grid boundary** (`MAP==0.600`,
  `head_readout_extraction_20260827.md` rail-condition table), which mechanically produces
  a small reported width regardless of how wrong the answer is. Its RMSE-to-truth
  (bias-dominated, **68× the floor**, ≈16% of h) is the honest number for that channel.
- In RMSE/h terms: 2D ≈ **9.5%** of h vs. a floor of **0.24%** of h; 1D ≈ **16%** of h.
  Both are an order of magnitude above the "useful" 5% line the F5 doc itself uses to
  separate informative from not (`docs/SIGMA_Z_SIGMA_M_FORECAST.md` §2), even though the
  *floor* sits deep in the "very useful" territory. **The gap is a misspecification/
  confusion signature, not evidence of an intrinsically starved dataset** — the theoretical
  information exists; the pipeline is not extracting it. Narrower-than-floor is the other
  named risk in the task brief; it does not apply here (measured is wider, not narrower).

---

## 4. Stop condition, stated numerically (item 4)

Given the floor is a **lower bound** (§5.1), the honest two-part criterion is:

- **Centering:** `|⟨h⟩ − 0.73| ≤ 3 × σ_h,floor(realistic σ_z, σ_M)` — a 3-floor-σ band is
  already the *most permissive* defensible band (the floor ignores every degradation
  channel that would only widen it further, never narrow it, so a genuine 3σ-of-the-floor
  miss cannot be explained away as floor-limited noise).
- **Width:** `σ_h,measured ≤ F × σ_h,floor`, `F` a dilution factor for the un-modelled
  impostor/multi-candidate competition — **not yet measurable from this node** (that is
  exactly what branch depth **B8.2**, "the two-channel calibration harness," is chartered
  to build: it needs the actual candidate-count-vs-z density, not a single-host toy). A
  provisional, deliberately generous placeholder `F=10` is used only to show the
  criterion is non-vacuous below.

**Current status, both venues, both channels: FAIL.** The width clause is borderline
even at the generous placeholder (`F=10`: 2D width/floor = 10.6, right at the line; F5
threshold-σ_M case would fail at F=10 by 43%). The centering clause fails **overwhelmingly
regardless of F** — a 38-to-68 floor-σ miss is not rescuable by any defensible dilution
factor. **This node's evidence therefore does not support "information-starved" as the
full explanation; it flags a bias/misspecification budget that must be found before a
width-only stop condition would mean anything** (this is a scope note for the tree, not a
new mechanism claim — the candidate mechanisms are already tracked under B1/B2/B4/B5/B6/B7).

---

## 5. Caveats (item 5)

1. **Single-host, no impostors (the headline caveat) — floor is a LOWER bound on
   achievable width.** Production's real numerator marginalizes over every catalogue
   galaxy in the localization volume, rate-weighted; this node assumes the true host is
   known with certainty. The measured/floor gap (§3) upper-bounds how much of it *could*
   be pure information starvation — it cannot itself be smaller than what impostor
   competition adds, but this node does not decompose the gap.
2. **Completion leg not modelled.** Median z=0.49 for this event set is well past GLADE's
   good-completeness regime (`docs/gates/G7_systematics_budget.md` row 15: binding at
   z≳0.3). For the *majority* of these 1588 events, "the single known host" is not even
   the right physical picture — a large fraction of the real production likelihood for
   these events is carried by the completion (out-of-catalogue) term, not a matched
   in-catalogue galaxy at all. The floor computed here should be read as a strict
   mathematical bound on the in-catalogue-host idealization, not a claim that most events
   have an identifiable host in reality.
3. **Route A's own failure is evidence, not just a discarded method.** §2.2/§2.3 show the
   single-host marginal posterior is itself highly non-Gaussian/degenerate at low z under
   GLADE photo-z (h–z compensation). Route B's Fisher floor is the correct *local/
   asymptotic* CRB, but Cramér–Rao is silent on how much worse the *actual* finite-sample
   posterior can be when the likelihood is this degenerate — and the June engine's own
   railed ~26–30% (N=400, synthetic) shows it can be dramatically worse, even before
   impostors. Two independent numbers (Route A's instability, the June engine's rail) both
   point the same direction.
4. **σ_z, σ_M are fixed population-representative constants, not per-event draws.** Real
   GLADE hosts have a mixed spec-z/photo-z flag population (spec-z ≈0.56% of GLADE per
   `docs/F4_SPECZ_DECOMPOSITION.md`) and a range of stellar-mass measurement quality; this
   node brackets the literature range (§0) rather than sampling a realistic per-galaxy
   distribution.
5. **Linear-Gaussian mass kernel** (shared with the F5 engine, `docs/SIGMA_Z_SIGMA_M_FORECAST.md`
   §4.2c) breaks down physically at σ_M≳1 (a factor-of-several uncertainty is not well
   described by a Gaussian) — irrelevant to this node's headline (no rescue at any
   realistic σ_M regardless of kernel shape), but would matter if σ_M were ever revisited.
6. **Real-production z-distribution ≠ June F5 toy population.** Median z=0.49 here vs.
   ≈0.15 in the June synthetic sweep, and this venue's horizon reaches z≈1.0 vs. the
   toy's 1.2 Gpc mock horizon. The June sweep's *quantitative* 2D/1D contour positions
   (e.g. "~50× tolerance", "2%/5% contours") were explicitly flagged there as population-
   shape-dependent (§4.6 of that doc, its own real-n(z) robustness pass); this node's
   floor is the first computed directly on the actual production z-distribution and N, and
   supersedes N-rescaling the June table for THIS venue's headline number — but the
   underlying real-n(z) qualitative finding (2-D gives no rescue at realistic σ_M) is
   reproduced here independently.
7. **z_i is recovered from the noisy measured `d_meas`, not the true injected z** (the
   local "raw" CRB is not present for seed61000; only `prepared_cramer_rao_bounds.csv`).
   This affects only the *location* used to evaluate `D'(z_i)` and the `(1+z_i)` mass
   anchor, not the injected `σ_dL,i`/`σ_Mz,i` values themselves; the effect on the floor is
   second-order and not expected to move the headline number, but was not separately
   quantified.
8. **This is a builder smoke-test, not a verified measurement** (standing rule 2). The
   numbers in §2.1 must be reproduced independently before the tree treats them as
   adjudicated.

---

## 6. Reproduction

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/b8_information_floor.py
```
Deterministic (no RNG) — every number in §2 is reproducible byte-for-byte from the pinned
CRB file (`seed61000/prepared_cramer_rao_bounds.csv`) and the literature constants in §0.
Runtime ≈1s, single core.
