# Handoff — Plotting overhaul follow-ups + CRB two-population mystery (2026-05-16)

## TL;DR

Phase A–H plotting overhaul shipped (commit `6dda72f`); 8 interactive HTMLs
deployed to GitHub Pages (commit `dc88a5f`). All 26 static + 8 interactive
figures now consume the canonical raw Σ log L_i loader and agree on
MAP = 0.7380 (discrete) / 0.7378 (continuous).

**The diagnostic plot raised a NEW question that wasn't resolved this
session:** the M_z-improvement HDI68-vs-N curve has a sharp elbow at N≈400–500
that turned out to reflect a **two-population structure in the CRB**, not
pure statistical convergence. The user hypothesised an M-vs-M_z simulation
bug; I cross-checked and **disproved** that specific hypothesis, but the
real cause (very likely two distinct injection campaigns merged) is still
not pinned down.

The single highest-value next-session action is: **identify what changed
between the Phase 45 (`run_phase45_20260501`) and seed300 extension
(`run_20260504_seed300_extension`) campaigns** that produced different
(M, z, d_L) sampling, and decide whether to keep them merged or split the
paper analysis.

---

## Current state (as of 2026-05-16T00:00 CEST)

### Code (all on `origin/main`, three new commits)

- `6dda72f` — Phase A–H plotting overhaul. Canonical raw Σ log L_i loader
  in `master_thesis_code/plotting/_helpers.py::load_canonical_combined_posterior`;
  replaces in-memory `Σ log(max(p,1e-300))` paths in `main.py` (fig01, fig02,
  fig15), `convergence_plots.py` (fig08 left panel), and
  `paper_figures.py::_load_combined_posterior`. Five new figures (fig16
  catalog, fig17 single-event, fig18 closure, fig19 info monotonicity,
  fig20 P_det surface) + 3 new Plotly interactives + validation harness
  (`master_thesis_code_test/plotting/test_canonical_map_consistency.py`,
  9 tests) + `Makefile` targets + `scripts/validate_figures.py`.
- `3de7817` — `docs/coauthor_meeting_2026_05_15.md`. Meeting overview of the
  H0 bias resolution story.
- `dc88a5f` — Refreshed `interactive/*.html` for GH Pages (5 updated, 3 new),
  plus index.html update with footer documenting the canonical strategy.

### Test suite
- 546 passed, 6 skipped, 0 failed (`uv run pytest -m "not gpu and not slow"`).
- ruff + mypy clean on `master_thesis_code/`.

### Data
- Production CRB at `simulations/cluster_run_production_h0p73_20260506/simulations/prepared_cramer_rao_bounds.csv`
  (1549 events; **contains the two-population structure**).
- Phase 48 1D and 2D posteriors at the same path's `posteriors{,_with_bh_mass}/`
  (63 h-values each). 2D posteriors are stripped of `galaxy_likelihoods`.
- Local 417-event dataset at `simulations/posteriors_with_bh_mass/`
  (full `galaxy_likelihoods` intact; used by fig17 single-event detail).
- Closure test posteriors at `simulations/closure_h{0p60,0p65,0p70,0p75}/posteriors/`
  (synced from cluster this session; 11/21/21/21 h-values respectively).
- Inference log for host-count parsing at
  `simulations/cluster_run_production_h0p73_20260506/simulations/logs/master_thesis_code_20260506_145714_h_0_73.log`.
- Host-count CSV at the same path's `diagnostics/host_counts.csv` (1473 events;
  median reduction 63.6%).

### Cluster
- `run_production_h0p73_20260506/simulations/` is the active production
  workspace. **2D posteriors there are 2.4 GB** — only stripped versions
  synced locally.
- Recent F4 commit `d1087f1` (Nadaraya-Watson p_det) is in upstream main but
  the production posteriors **were generated against the F1 estimator**, not
  F4. The canonical loader cache will use whatever's in those JSON files —
  no automatic regeneration. Per the Phase 49 handoff, F4 is meant to close
  the spiky-posterior issue; if F4 needs a re-run, the canonical cache
  `posteriors/canonical_combined.json` will need a `refresh=True` re-evaluation.

### Docs
- `docs/coauthor_meeting_2026_05_15.md` — committed, ready for the meeting.
- `docs/H0_BIAS_RESOLUTION.md` — **NOT yet updated** with Phase A unification
  entry. Should add a §3.16 (or §3.X numbered next) entry documenting:
  the silent divergence bug, the canonical loader, the validation harness.
- `.planning/MILESTONES.md` — **NOT yet updated**; the v2.1 plotting milestone
  may need a row.
- `interactive/index.html` — updated this session with three new figure entries
  and footer note about the canonical strategy.

### Commits unpushed
- None — all three commits pushed to `origin/main` this session.

---

## The CRB two-population mystery

**The user's hypothesis (M vs M_z simulation mismatch) was DISPROVED**
quantitatively. See the cross-check in the chat log; key numbers:

| quantity | rows 0–400 | rows 400+ | residual after (1+z) correction |
|---|---|---|---|
| median M (M☉) | 3.7×10⁵ | 2.2×10⁵ | factor 1.51 |
| median d_L (Gpc) | 0.32 | 0.07 | n/a |
| median SNR | 24 | 31 | n/a |
| median z (estimated) | 0.074 | 0.017 | n/a |
| n hosts with M_z cut (median) | 76 | 2 | n/a |
| unique GLADE+ host indices | 272 | 549 | 35 overlap (13%) |

Pure (1+z) re-labelling cannot account for the 51% residual gap in M after
correction. The M values themselves cluster on a **discrete library**
(M1 rate-table samples) and the distribution of which library entries get
drawn changed wholesale across the boundary — three M-values
(2.23×10⁵, 8.5×10⁴, 1.08×10⁵) appear almost exclusively post-boundary;
one (3.18×10⁵) appears mostly pre-boundary.

The most plausible explanation is: **the production CRB is a concatenation
of two injection campaigns** (Phase 45 + seed300 extension) that sampled
the (M, z) prior differently. The historical M_z fix the user remembered
dates from **July 2024** (commits `b08b768` / `8533ae5`), well before
either of these 2026 campaigns — so it is not the cause.

**Bootstrap subsampling is random** (`rng.choice(n_common, size=n_sub,
replace=False)` in `convergence_analysis.py:508`), so the elbow at N=400–500
in paper_m_z_improvement.pdf is NOT because events are picked in time order.
The elbow is real: at low N the wide multi-modal L(h) from population A
(many candidate hosts) drowns out population B; above N≈500 the sharp
unique-host L(h) from population B reinforce statistically and the joint
posterior collapses to the truth.

Note also: 2D HDI68 at N=1400 = **0.001 = grid resolution floor**. The
HDI saturates because the dense core of the h-grid (in [0.710, 0.750])
has Δh=0.001 — the true 2D posterior is narrower than measurable.

---

## Action items (ordered by priority)

### A. ~~Pin down what differs between Phase 45 and seed300 extension~~ — RESOLVED 2026-05-16

**Resolution:** the boundary at row 424 in `prepared_cramer_rao_bounds.csv` is
the **seed200/seed300 concatenation seam**, not a code-difference between
campaigns. Verified on cluster:

- `run_20260401_seed200/simulations/cramer_rao_bounds.csv` (SNR_THR=15, 4497
  raw events) → SNR≥20 subset = **424 events**, top-5 M = {463534 (×176),
  318738 (52), 294877 (43), 342686 (39), 271107 (29)}. Matches production
  rows 0–424 to the unit.
- `run_20260504_seed300_extension/simulations/cramer_rao_bounds_simulation_*.csv`
  (SNR_THR=20, all SNR≥20) → **1050 events**, top-5 M = {223872 (379),
  463534 (317), 85539 (110), 108123 (51), 63238 (49)}. Matches production
  rows 424+ closely; ~75 events unaccounted (likely small third extension).

`git log -- master_thesis_code/cosmological_model.py` shows **no sampler
changes** between Apr 7 and May 4. PE-01 (55a6d99) threads `h_inj` but is a
no-op at h_inj=H=0.73 by construction.

**Mechanism**: per-task emcee chains in `Model1CrossCheck.setup_emri_events_sampler`
(nwalkers=20, burn_in=1000) **under-mix** the M1 mass prior. Each task
converges to a seed-dependent sub-region of (M, z). Verified by per-task
M-library inspection: within seed300 alone, tasks 10 & 20 are dominated by
M≈4.6e5 while tasks 2, 5, 30, 40, 49 are dominated by M≈2.2e5. Same drift
in seed200's surviving per-task survivors.

**Impact on H0**: none. The d_L–z relation per event and event-by-event
likelihoods are unaffected by the M-marginal heterogeneity. The bootstrap
σ_boot is correct; the convergence-curve elbow at N≈420 is a concatenation-
order artifact, NOT "data became more informative."

**Paper policy adopted**: keep merged 1549-event CRB. Disclose in
methods/appendix that the production CRB is `seed200(SNR≥20) ⊕
seed300_extension(SNR≥20)` with concatenation in that order; document the
per-task emcee under-mixing observation; show H0 MAP robustness across
(merged, seed200-only, seed300-only) subsets.

**Optional fix** (not blocking paper): increase `burn_in_steps=1000 → 10000`
and `nwalkers=20 → 50` in `cosmological_model.py:setup_emri_events_sampler`.
One-line config change, requires re-running CRB campaign on GPU. Only worth
doing if reviewers flag the M-prior heterogeneity.

Memory entry: `memory/project_crb_two_population.md`.

---

### A-bis. Original notes (superseded; kept for traceability)

**Goal (original):** explain the CRB two-population structure, decide whether to keep
the campaigns merged or split the analysis into A-only/B-only subsets.

**Concrete steps:**
1. On cluster, read `run_metadata.json` in both runs:
   ```
   ssh bwunicluster cat /pfs/work9/workspace/scratch/st_ac147838-emri/run_phase45_20260501/run_metadata.json
   ssh bwunicluster cat /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260504_seed300_extension/run_metadata.json
   ```
   Compare seeds, h_value CLI arg, simulation_steps, any cosmological-model
   args, and `git_commit` of each run.
2. `git diff` the simulation-touching files between the two `git_commit` values
   — specifically `master_thesis_code/cosmological_model.py`,
   `master_thesis_code/datamodels/parameter_space.py`,
   `master_thesis_code/M1_model_extracted_data/`,
   `master_thesis_code/galaxy_catalogue/handler.py`.
3. Inspect the M-value distribution in EACH per-task CSV
   (`cramer_rao_bounds_simulation_{0..49}.csv`) — is the (M, z) sampling
   uniform within each task and only the BATCH differs, or did individual
   tasks of one campaign get a different prior?
4. If the campaigns genuinely used different priors → **decide the paper
   policy**: (a) keep merged (the published result is on a heterogeneous
   detection set, which is fine if disclosed), or (b) split A/B and quote
   the BH-mass channel gain on the homogeneous B subset where it's not
   masked by population A.

**Files to read first:**
- `master_thesis_code/cosmological_model.py::Model1CrossCheck.sample()`
- `master_thesis_code/datamodels/parameter_space.py::ParameterSpace.randomize_parameters()`
- `cluster/evaluate_production_h0p73_dense.sbatch` and the seed300
  extension's sbatch (to see CLI flags)

### B. Refine the h-grid dense core to break the HDI68 floor (medium — visual paper)

**Goal:** the M_z-improvement bottom panel currently saturates at HDI=0.001
because that's Δh_min. Refining the dense core to Δh=0.0005 in [0.720, 0.740]
would let the 2D channel's true tightening show.

**Concrete steps:**
1. Cluster sbatch with refined h-grid: keep wings at Δh=0.010 but use
   Δh=0.0005 in [0.720, 0.740] (20 dense points instead of the current
   ~30 at Δh=0.001 across [0.710, 0.750]).
2. Run only the with-bh-mass channel on the new grid (1D doesn't need
   sub-mille resolution at N=1473).
3. Re-generate the canonical posterior cache by deleting
   `posteriors_with_bh_mass/canonical_combined.json` and calling
   `make validate-figures`.
4. Re-run `make regen-figures` to refresh fig17, fig18, paper_m_z_improvement.

**Cost:** ~30–60 min cluster time at the dense-only resolution (the wings
already exist).

### C. Sync the full unstripped 2D posteriors for single-event detail (low — backlog)

**Goal:** fig17_single_event_detail currently falls back to the local
417-event dataset because cluster 2D posteriors were stripped of
`galaxy_likelihoods` to save 2.4 GB. To pick representative events from
the production 1473-event dataset, we need the full data.

**Two options:**
1. **Cheap:** strip script on cluster that keeps `galaxy_likelihoods` only
   for 3–5 hand-picked event IDs (~5–10 MB to sync). Mentioned in the
   plotting plan, never executed.
2. **Expensive:** rsync the full 2.4 GB. Better long-term but slow.

**Recommended:** option 1. Pick 3 events at FWHM percentiles 25/50/75 from
the production posterior, write a one-liner Python script on the cluster
that strips everything else, rsync the result to
`simulations/cluster_run_production_h0p73_20260506/simulations/posteriors_with_bh_mass_detail/`,
update `_gen_single_event_detail` in `main.py` to prefer this dir.

### D. Instrument host-count persistence properly (low — backlog)

**Goal:** Phase B parses host counts from inference logs (works, but lossy
if logs rotate). Add a structured CSV write inside
`BayesianStatistics.evaluate` so future cluster runs persist this directly.

**Note:** this touches an inference-path file (`bayesian_statistics.py`),
which is gated by the **physics-change protocol** in CLAUDE.md. Even though
the change is purely diagnostic (no formula touched), confirm with `/physics-change`
before editing to avoid CI gates flagging it.

**Files to change:**
- `master_thesis_code/bayesian_inference/bayesian_statistics.py::evaluate`
  — append a row to `diagnostics/host_counts.csv` for each event after
  `get_possible_hosts_from_ball_tree` returns.

### E. Document Phase A unification in H0_BIAS_RESOLUTION.md (low — docs hygiene)

**Goal:** the bias resolution doc catalogue (§3.x) should have an entry for
the silent figure-divergence bug found 2026-05-15, the canonical loader fix,
and the validation harness. Currently only the chat log records it.

**Suggested section heading:** "§3.16 Silent figure-MAP divergence across
three combination paths (Phase A unification)". Symptom / Mechanism /
Diagnostic / Fix / Evidence / Limitations / Reference, same format as
existing §3.x entries.

### F. Decide if F4 (Nadaraya-Watson p_det) needs a posterior re-run (medium)

**Goal:** the production posteriors used in the canonical loader were
generated against the F1 estimator (commit `87ea7a8`). F4 (`d1087f1`,
Nadaraya-Watson kernel) is now upstream and is expected to fix the spiky
posterior. The current Phase A canonical cache will silently keep using
the F1-era posteriors until the per-h JSONs are regenerated.

**Concrete steps:**
1. Check Phase 49 handoff (`.planning/HANDOFF-PHASE49-MECHANISM-VERIFY-20260514.md`)
   to confirm F4's expected effect on the production posterior.
2. If F4 is expected to change MAP/HDI68 materially → re-submit the
   production sbatch with F4 + delete
   `posteriors/canonical_combined.json` + run `make validate-figures` to
   re-cache.
3. If F4 is a no-op at h=0.73 in practice (or the change is sub-σ_boot) →
   document this in `docs/H0_BIAS_RESOLUTION.md` and leave the Phase 48
   posteriors as the paper reference.

---

## Verification commands (next session can sanity-check current state)

```bash
# 1. Confirm tests still pass after pulling new upstream
uv run pytest master_thesis_code_test/ -m "not gpu and not slow" -q

# 2. Confirm the canonical loader produces the expected MAP
make validate-figures
# expect: posteriors discrete=0.7380, continuous=0.7378
#         posteriors_with_bh_mass discrete=0.7380, continuous=0.7378

# 3. Regenerate the 26 static figures
make regen-figures

# 4. Regenerate the 8 interactive HTMLs
make regen-interactives

# 5. Check GH Pages deploy succeeded
gh run list --limit 3 --json status,conclusion,name
# (and visit https://jasperseehofer.github.io/MasterThesisCode/interactive/)
```

---

## Cross-project lessons worth recording (for /wiki-debrief)

1. **Convention drift across "same" computation paths.** When a quantity
   is computed by multiple code paths that should agree, add a single
   validation test that asserts pairwise equality on synthetic data.
   The Phase H harness is the template
   (`tests/plotting/test_canonical_map_consistency.py`).

2. **Random subsampling masks heterogeneous-population structure.**
   When the bootstrap is random, an elbow in a convergence curve is NOT
   evidence of "data added late" — it can mean the population itself is
   bi-modal in informativeness. Always inspect the per-event distribution
   of the relevant scalar (here: `n_with_mass`) before attributing curve
   features to N alone.

3. **Disprove with the explicit factor.** The user's M/M_z hypothesis was
   testable: if pre-boundary M = M_z = M_source(1+z), then dividing by
   (1+z_estimated) should match post-boundary M. It didn't (residual factor
   1.51), so the hypothesis fails quantitatively. Quick numerical
   counter-checks beat narrative debate.
