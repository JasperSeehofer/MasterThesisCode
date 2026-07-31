# ch07_FLAGS.md — disagreements found while building Chapter 7

Raised by the ch07 agent, 2026-07-31. Per `BOOK_DESIGN.md` §4.1 and the fan-out
brief: where a generator's recomputation disagrees with the build spec or a cited
document, the thread is stopped and recorded. **Nothing below has been reconciled
in either direction.** Both values are carried into `data/ch07_*.json` and both are
visible on the page.

---

## FLAG-1 — σ_dL/dL of EMRI-889 (the running example's dossier row)

| source | value |
|---|---|
| `BOOK_DESIGN.md` §1, Ch 1 card ("EMRI-889 … σ_dL/dL = 8.0×10⁻⁵") and Ch 6 card ("the dossier gains σ_dL/dL = 8.0×10⁻⁵") | **8.0 × 10⁻⁵** |
| `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`, row 889: `sqrt(delta_luminosity_distance_delta_luminosity_distance)/luminosity_distance` | **8.983 × 10⁻⁴** |

Details of the recomputation (ch07 agent, `gen_ch07.py` assertion + manual check):

* Row 889 matches the spec's other identifiers exactly — `M = 724631.5 M☉`
  (spec: 7.25×10⁵), `mu = 10`, `luminosity_distance = 0.0888792 Gpc` (spec: 88.9 Mpc),
  `SNR = 1424.72` (spec: 1425), `host_galaxy_index = 859360` (spec: 859360),
  `in_catalog = True`. So the **event identification is not in doubt**; only the
  quoted distance precision is.
* The value is identical in all six seed61000 CRB copies
  (`seed61000/` and `real_r1..r5/`), i.e. it is not a run-to-run difference.
* Marginal and sky-conditional widths agree to 4 digits
  (8.9833×10⁻⁴ vs 8.9786×10⁻⁴), so the discrepancy is **not** a
  marginal-vs-conditional confusion.
* The independent Gate-B artifact
  `gate_b_20260730/c7_kernel_measure_results.json` stores, for the same host,
  `rel_dL = 8.983284×10⁻⁴` and `sigma_frac_cond = 8.978594×10⁻⁴` — i.e. the
  project's own C7 driver reads the same 8.98×10⁻⁴.
* The median over the 76 in-catalogue hosts is `rel_dL = 5.30×10⁻³`, consistent
  with the ratified `σ_dL/d_L = 0.54 %` golden-set figure in
  `docs/derivations/hostz_pv_photoz_kernel.md` §0 — another sign that the CSV
  scale is the right one.

**Downstream consequence (for the ch01/ch06 agents and the integrator):**
`BOOK_PEDAGOGY.md` Q6.5's answer computes "roughly **6000×** larger" from the
8×10⁻⁵ figure. With 8.98×10⁻⁴ the same comparison gives ≈ 5.5×10² at
σ_z/z = 0.49. Chapter 7 therefore **does not use the 6000× number anywhere**; it
states the comparison from artifacts it can chip (median σ_dL/dL = 0.53 % of this
venue's 76 hosts vs the ratified PV/photo-z dominance measurements of
`hostz_pv_photoz_kernel.md` §0). Ch 1 / Ch 6 should decide the dossier value; Ch 7
prints the CRB value with this flag attached.

---

## FLAG-2 — the C7 rail threshold σ_z/z > 0.256

| source | value |
|---|---|
| `gate_b_20260730/C7_README.md` §1, `CLAIM_2D_BIAS_20260730.md` C7 (as amended 2026-07-30), `gate_b_20260730/ADJUDICATION_20260730.md:148` and `:460` | **0.256** |
| Solving those same documents' **corrected law** `h_eff/h_true = [1+√(1+12ε²)]/2` for `h_eff = 0.86` at `h_true = 0.73` | **0.2644** |

* The threshold is stated identically in three artifacts, so it is not a
  transcription slip inside one file.
* It is **not** computed anywhere in `c7_kernel_measure.py` — the driver only
  records `frac_peak_above_086`; the threshold was evaluated by hand in the README.
* The delivered per-host measurement brackets the same crossing: median peak
  `0.8476` at ε = 0.25 and `0.9390` at ε = 0.35, which interpolates to
  ε ≈ 0.264 for the 0.86 edge — i.e. the **measurement agrees with 0.2644**, not
  with 0.256.
* Neither the superseded 8ε² form (which gives 0.324) nor the small-ε limit 3ε²
  (which gives 0.244) reproduces 0.256 either, so the origin of 0.256 could not be
  reconstructed.

**How Chapter 7 handles it:** the page prints the artifacts' **0.256** as the rail
threshold (that is what the adjudication says, and the chapter may not resolve what
the project has not resolved), draws the reader's live crossing where the quoted law
actually puts it, and carries a visible provenance note stating both numbers and
pointing here. `data/ch07_c7.json → rail_threshold` carries
`artifact_value: 0.256` and `recomputed_from_quoted_law: 0.2644` side by side.

---

## Non-flags — checks that PASSED (recorded so they are not re-run)

These were verified at generation time by `gen_ch07.py` and are reported in its
stdout; they are listed here because a reviewer may otherwise assume they were
taken on trust.

| gate | what was re-measured | result |
|---|---|---|
| G1 | `G2b:288-293` exact posterior-mean shift table at z_g = 0.05 (Ω_m = 0.3) | reproduced to 0.1–0.4 % |
| G2 | `G2b:253-259` amplitude table `C(z̄) = h·s·dln f/dz` | reproduced to < 1 % (only `dln f/dz(0.30)` differs, 3.84 vs the quoted 3.85 — rounding) |
| G3 | the corrected C7 law `[1+√(1+12ε²)]/2` against the delivered per-host median peaks, ε ≤ 0.49 | worst \|Δ\| = 0.0032 (C7_README claims "< 1 % up to ε = 0.5") |
| G4 | the observed in-catalogue ball-numerator tilt, recomputed from `real_r1/diagnostics/event_likelihoods.csv` + `Δln Σ_glob = +0.027597` | max \|Δ\| vs the stored `observed_incat_dln` = **7.2×10⁻¹⁶**; median +0.3082, 93.24 % positive |
| — | `C7_README` §3 quartiles of the indicative σ_z/z over the 76 hosts (0.379 / 0.519 / 0.644) | reproduced exactly from the delivered JSON |
| — | `C7_README` §3 "98.7 % of hosts peak above 0.86" at indicative widths | reproduced exactly (75/76) |

Note on G4: the `diagnostics/` directory is untracked churn and is absent from the
book worktree; `gen_ch07.py` looks in a sibling checkout (relative path only) and
degrades to SKIPPED rather than failing when it is unavailable. The JSON never
depends on it.

---

# REVISION — 2026-07-31 (post-review pass, `REVISION_WORKLIST.md` §C-ch07)

Appended, not rewritten: everything above is the historical record as raised
during the build. This section records what the revision pass changed and why.

## FLAG-1 — **RESOLVED** by author mandate (worklist §A-D1)

The author's D1 decision supersedes the "carry both values" disposition recorded
above. The six-chapter measured value is adopted as the spec value book-wide:

> **σ_dL = 7.98×10⁻⁵ Gpc (absolute) · σ_dL/d_L = 8.98×10⁻⁴ (fractional).**
> The old spec figure `8.0×10⁻⁵` was the *absolute* σ_dL in Gpc carried under a
> *fractional* label — a ×11.25 slip, not a disagreement about the event.

Everything recorded under FLAG-1 above (row identity confirmed on M, μ, d_L, SNR
and host index; identical in all six seed61000 copies; marginal-vs-conditional
ruled out at 4 digits; the project's own C7 driver storing the same `rel_dL`) is
exactly the evidence the mandate acted on — the flag was right, and it is now
closed rather than withdrawn.

Applied on ch07:
- Dossier card now carries the **canonical** D1 row
  (`d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴`, `BOOK_CANON.sigmaDL.dossierRow`);
  SNR and host index moved to their own row. The fractional value is still
  filled live from the CRB by the page script, and is now also pre-rendered so
  the static reader sees it.
- The boxed FLAG-1 provenance note became a **one-line erratum note** (`.ch07-note`,
  carrying `BOOK_CANON.sigmaDL.erratum` verbatim as its opening sentence), keeping
  ch07's own arithmetic (σ_dL = 7.98×10⁻⁵ Gpc against d_L = 0.0889 Gpc) and the
  ≈550× downstream correction. It is no longer framed as an open dispute.
- Provenance panel: FLAG-1 listed as RESOLVED; FLAG-2 still OPEN.
- `gen_ch07.py` / `ch07_c7.json`: the bare `rel_dL_build_spec_value` and
  `rel_dL_flag` keys are gone. `event_889` now carries `sigma_dL_Gpc`
  (**measured**, `7.9843e-05`, computed as `rel_dL × d_L` with d_L read from the
  tracked seed61000 CRB by the new `_d_l_889_gpc()` helper), `d_L_Gpc`, and a
  single `rel_dL_erratum` string. The retired figure now exists on this page and
  in this data file **only inside erratum text** — `qa_gates.py` gate D1 passes.

## FLAG-2 — unchanged, still OPEN

The 0.256 vs 0.2644 rail-threshold disagreement is untouched by this pass and by
cell B (cell B measures the *magnitude*, not the threshold's provenance). Expert
B's independent solve reproduced 0.2644 and the ≈0.264 interpolation, and the
review's PRAISE section asks for the disposition and its closing sentence to be
kept verbatim — they are. Worklist §F-6 routes the archaeology to the author.

## NEW — the C7 decider landed: **the 2×2 cell B**, 2026-07-31

`CELLB_READOUT_20260731.md`, evaluate **6103219** / combine **6103220** (the
resubmission of the pre-registered 6101146/6101147 after a pure-plumbing symlink
failure; test design and pre-registration unchanged; code `7fd60bb`, the same
commit as cells A and C).

Numbers now carried by ch07 (§6 landed block, I7.2 noscript, `ch07_c7.json →
hosts.resolved_by_cellB` and `conflict.decider`):

| read | cell B (unscattered) | C (#53 r1, scattered) | A (#51 estimator) |
|---|---|---|---|
| catalogue-leg per-event argmax at prior top | **90.7%** (68/75) | 89.2% (66/74) | — |
| combined per-event in-cat argmax at 0.86 | 69.7% | 57.9% | 5.3% |
| in-catalogue **class** argmax | **0.860** (as registered) | 0.860 | — |

**Scope, stated on the page and binding:** cell B settles C7's **magnitude and
attribution**, *not* the G2b↔C7 collision — a derivation-level conflict no
posterior can settle. G2b's CONFIRMED verdict is untouched. The fix stays
author-gated, must **explicitly supersede G2b**, and — new constraint from the
readout (§Next steps 1b) — **must not be the historically-exonerated "p_det
inside the numerator alone" form**. Chapter 7 still may not say "the kernel is
wrong, settled".

**Honest-staleness nuance carried on the page** (this chapter's own flag culture
applied to itself): the *indicative* local `z_error` column implied 75/76 =
98.7% of hosts peaking above 0.86; the staleness-free measurement gives 90.7%.
These are different statistics (reconstructed unclipped single-host peak vs
delivered clipped `L_cat` argmax), so it is not a contradiction — but the page
says *"the staleness caveat resolves in the confirming direction, with the
delivered rail somewhat weaker than the stale column implied"*, never
"98.7% confirmed".

**Naming (worklist D3 / expB MJ-3):** the 2026-07-31 object is called
**"the 2×2 cell B"** throughout. §5 gained a scoping paragraph so ledger #88's
per-leg **85.3% / 86.7%** (seed1000 deep venue, δ-kernel alone) and the 2×2 cell
B's **72%** (this campaign's venue, whole estimator configuration) cannot be read
as a disagreement — and it notes that ledger #88's own "Cell B" is a different
object, which `BIAS_HISTORY_LEDGER.md` §3 relabels A′.

## Other revision items applied to ch07

- **Rail pip** is now the canonical `BOOK_CANON.cellB` pip
  (`cell B (2026-07-31): estimator owns +0.060 of the 2D +0.083`), read from
  `js/manifest.js` at runtime with a greppable literal fallback.
- **φ_cat defined at first use** (§6, tomas-M4): the catalogue's own selected
  number density, `φ_cat(z) ∝ f(z,Ω)·(dV_c/dz)/(1+z)`, as against `w_pop`'s prior
  over where an EMRI host is — the two priors are over two different random
  variables. Tagged `data-term="phcat"` (the key exists in `Book.SYMBOLS`). One
  added clause records that standard-practice kernels (Laghi 2021 / Turski 2023 /
  gwcosmo) do **not** deconvolve at all, which is what makes deconvolution this
  project's own deviation **D8** and therefore the thing under test. **This names
  the axis; it adjudicates nothing.**
- **Trap relocation (ped-M3).** Trap 7.A moved from the page bottom to §2,
  immediately after the coverage-collapse paragraph — the misconception forms in
  the cold open and is dismantled there, so the trap now fires while it is
  forming rather than ~40 minutes later. Trap 7.B moved to sit immediately
  *before* §6's twist (end of §5, after the stage-3 unlock button), which is the
  sentence the reader thinks entering §6; its closing clause now reads
  "…pre-registered to decide, and did, on 2026-07-31".
- **Deck de-spoiled (ped-B1 tail, D4):** subtitle and `<meta description>` now
  end on *"the central value is not the safe choice"* instead of *"does not blur
  the answer — it moves it"*, which collapsed `#ch07-predict-1` from a 3-way to a
  2-way.
- **Q7.1 → transfer form** (ped-M2): no longer a verbatim re-ask of the cold-open
  predict. Now asks what feature of the volume prior sets the **sign**, and what
  would have to be true of the prior for symmetric widening to be right (answer:
  a flat prior, `s = 0`). Measured 5-gram overlap with the body **50% → 4.3%**.
- **Q7.5 → transfer form** (ped-M1): now asks for the two conditions a
  "keep only the clean data" rescue must satisfy, which one this cut fails, and
  what change to the *world* would repair it. Overlap **62% → 6.3%**.
- **Q7.6 + the closing section: σ_Mz both-values (D5, tomas-B3).** Both sites now
  carry the claim file's `σ_Mz/M_z ≈ 10⁻⁴` chipped `CLAIM C4` **and** the measured
  median `8.8×10⁻⁸` (889: `1.36×10⁻⁹`) with the `F-ch06-5` pointer. No silent
  substitution; Q7.6's answer notes explicitly why the disagreement does not move
  its conclusion (the binding width is the *catalogue's* mass, not the GW's).
- **I7.2 noscript fallback** (expB MN-2) gained the cell-B resolution with the
  same numbers, so a static reader no longer receives the stale-column prediction
  with no answer.
- **Closing section** ("What this leaves on the table") no longer says the
  deciding experiment is "not yet run"; it says what landed, what that closed,
  and what it deliberately did not.

## Decisions recorded (worklist §G-6 scoping)

- **Q7.4 kept as-is.** Measured 5-gram overlap 27.7%, just over the 22% line, but
  Q7.4 is not among the worklist's named offenders (only Q7.1 and Q7.5 are), and
  §G-6 explicitly rejects a mechanical rewrite of the full question set. Q7.4 is
  also the only question that carries the C5 fair-framing amendment in both
  halves, which the review's PRAISE section (PR-4) singles out — re-aiming it
  would put that binding amendment at risk for a 5.7-point overlap gain.
- **`\varphi_{\rm cat}` vs the passport's `\phi_{\rm cat}`.** The page keeps
  `\varphi`, because the adjudicated C7 quote it is defined against already uses
  `\varphi` and that box is quoted text. Flagged here for the integrator in case
  the passport card should be re-glyphed for consistency; purely cosmetic.
- **86.7% (ledger #88) and 72% (the 2×2 cell B) both kept**, scoped rather than
  reconciled — they measure different decompositions on different venues, and
  picking one would be exactly the tidying this chapter refuses elsewhere.
