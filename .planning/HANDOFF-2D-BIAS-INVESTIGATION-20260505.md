# Handoff — 2D channel residual bias investigation (2026-05-05 11:55)

**Audience:** a fresh Claude Code session picking this up cold.
**Author:** the morning session that watched the multi-truth panel run.
**Status of caller:** the 7-truth panel is still running on `dev_cpu_il`.
This handoff is for an investigation into a **post-Tier-3 residual bias**
that the partial 4-truth panel surfaced — **don't touch the running panel.**

---

## TL;DR

The Tier 3 D(h) double-counting fix (committed 2026-05-04, `6754ddb`) ate
the worst of the H₀ MAP bias at h=0.73 — both 1D and 2D PASSED on Phase 45's
424-event sample (z=+1.4σ and z=+1.97σ). Tonight we re-ran the multi-truth
panel on the **phase46-merged CRB (1549 SNR≥20 events = Phase 45 + seed=300
extension)**. The partial 4-truth panel verdicts:

- **1D channel**: z_panel=4.1, χ²_red=7.0 → `verdict_mean=FAIL`, marginal
- **2D channel**: z_panel=**65**, χ²_red=**1471** → `verdict_mean=FAIL`,
  spectacular

What changed between the closure-test PASS and tonight's panel FAIL: only
the event count (424 → 1549). The absolute bias didn't shrink — but σ_boot
tightened by ~1.9× (√(1549/424)). Bias that read as 1–2σ at small N now
reads as 4–55σ at large N. **This is a structural residual** that the Tier
3 fix did not fully eliminate, and it's much louder in the 2D channel than
in the 1D.

Investigation goal: isolate the residual mechanism in the 2D channel, fix
it, re-validate on the panel data.

---

## State of the running panel — DO NOT TOUCH

- **Orchestrator** PID `834247`, log `.claude/runlogs/full_devcpuil_20260504_221932.log`.
- **Currently on truth 5/7 (h=0.75)**. Sbatch `4227491_*` running.
- **ETA full panel:** ~15:00 today (3 truths × ~80 min/truth, including the
  Slurm inter-batch gap on dev_cpu_il QOS=dev limit of 1–2 concurrent).
- When the orchestrator hits all 7, it auto-runs `test_24` and writes the
  full panel verdict to `scripts/bias_investigation/outputs/phase45/multi_truth_sweep.{json,png}`
  (overwriting the partial-4-truth content there now — but the snapshot
  `multi_truth_sweep_partial4truths_20260505_115451.{json,png}` is preserved).

**Don't kill, restart, or otherwise interfere.** The investigation can
start without the remaining 3 truths.

---

## Partial verdict numbers (4 truths)

Source: `scripts/bias_investigation/outputs/phase45/multi_truth_sweep_partial4truths_20260505_115451.json`.

| h_truth | N | 1D MAP | 1D bias | 1D σ_boot | 1D z | 2D MAP | 2D bias | 2D σ_boot | 2D z |
|---------|---|--------|---------|-----------|------|--------|---------|-----------|------|
| 0.60 | 903 | 0.6044 | +0.0044 | 0.0011 | +3.83 | 0.6149 | +0.0149 | 0.0003 | **+54.98** |
| 0.65 | 1019 | 0.6560 | +0.0060 | 0.0030 | +2.02 | 0.6570 | +0.0070 | 0.0017 | +4.03 |
| 0.70 | 1265 | 0.7067 | +0.0067 | 0.0049 | +1.37 | 0.6960 | −0.0040 | 0.0152 | −0.26 |
| 0.73 | 1473 | 0.7279 | −0.0021 | 0.0033 | −0.64 | 0.7512 | +0.0212 | 0.0006 | **+37.08** |

**Per-event pos_frac**: 1D mean=0.69 std=0.05; 2D mean=0.75 std=0.09. Both
PASS the `verdict_shared_injection_pull` heuristic (std<0.05 cutoff is the
flag), but the 2D mean=0.75 is far from 0.5 — events systematically pull
their per-event MAP above truth in 2D more than in 1D.

**Sign concordance**: 3/4 truths positive in both channels (p=0.62 binomial,
not significant). Only h=0.70 in 2D goes very slightly negative.

---

## Hypotheses (ordered by promise)

### H1 — σ_boot underestimates the seed-dependent MAP drift
*See memory `finding_seed_dependent_map.md`.* At h=0.73 we already know the
production MAP is seed-dependent at ~0.02 scale (0.7400 → 0.7233 with a
fresh `prepare_detections` seed). σ_boot from event resampling at fixed
truth doesn't capture this; it captures within-realization scatter only.
**If H1 is correct,** the "true" σ that should appear in the denominator is
~0.02 instead of 0.0006, which would move the 2D h=0.73 z from +37 to ≈+1
— PASS.

**Diagnostic:** rerun `prepare_detections` on the merged CRB at h=0.73 with
a different seed (say 211 instead of 204), recompute the per-event likelihoods
and the combined MAP. Repeat with a third seed. The realization-to-realization
scatter of the 2D MAP is the right denominator. If it's >> σ_boot, that's H1
confirmed.

This is an inexpensive diagnostic and the natural first move.

### H2 — residual D(h) coupling specific to the 2D joint posterior
*See §3.13 of `docs/H0_BIAS_RESOLUTION.md` for the Tier 3 fix.* The fix
removed the outer −N · log D(h) from `combine_log_space`. Per-event L_comp
= num/D was preserved (it's the prior normalization per Gray Eq. 31). But
the 2D channel uses a different per-event likelihood (`posteriors_with_bh_mass/`
JSONs come from a slightly different code path — joint position-and-mass
likelihood). It's possible the 2D L_comp's D(h) normalization is wrong in
a way Phase 32 didn't fix.

**Diagnostic:**
- `scripts/bias_investigation/test_22_dh_double_count.py` (joint MAP with
  outer correction coefficient c ∈ {0, 1}) was the Tier 3 confirmer. Run it
  on the partial-panel posteriors to see if c=0 vs c=1 still differ for
  the 2D channel.
- Inspect `master_thesis_code/bayesian_inference/posterior_combination.py`
  and trace where the 2D channel's L_comp is built. Compare per-h-value
  D(h) against `posteriors/` (1D) — if they differ, suspect.

### H3 — BH-mass channel D(h)/normalization mismatch
The 2D analysis multiplies the position likelihood by a redshifted-MBH-mass
likelihood (M_z). The galaxy catalog provides a mass distribution; the EMRI
provides M_z = M (1+z). At each h-value, M_z → M depends on z(d_L; h), so
z(d_L) shifts with h, which shifts the M-channel likelihood. If this h-dependence
is computed inconsistently between per-event and combine, you get a 2D-only
bias.

**Diagnostic:** in `paper_figures.py` look at `_load_per_event_with_mass_scalars`
or `convergence_analysis._load_per_event_with_mass_scalars`. Compare against
the 1D loader. Also: trace M-channel D(h) inside the per-event 2D evaluation
in `bayesian_inference/bayesian_statistics.py` (single-event 2D likelihood).

### H4 — shared-injection-set pull through correlated host-galaxy draws
The `verdict_shared_injection_pull` flag is conservative (it FLAGs only if
pos_frac mean ≠ 0.5 *and* std<0.05). Tonight's 2D pos_frac mean=0.75 std=0.09
PASSES the flag (std=0.09 > 0.05) but the **mean** of 0.75 is suspicious —
across 4 truths, 75% of per-event posteriors lean above their truth. With
21 h-values per truth and 4 different truths, that consistent skew implies
the same injections (same sky positions, masses, redshifts) systematically
pull MAP above their truth.

**Diagnostic:** injection-set bootstrap (resample CRB rows before rescaling,
repeat the panel). ~10× compute cost — out of scope for a quick investigation,
file as follow-up if H1–H3 don't resolve it.

### H5 — fine-grid extremes (0.60, 0.85) are clamped and that distorts MAP location
At h_true=0.60 the LamCDM prior clamp gives 11 grid points 0.6000–0.6500
(asymmetric: only positive bias direction is sampled). The MAP is forced
to be ≥ truth. **This explains why h=0.60's bias is always positive but
not why σ_boot is 0.0003 (much tighter than other truths).** Worth checking
if the parabolic-refine step on a clamped grid does something pathological
to σ_boot estimation.

---

## Where to start (concrete first hour)

1. **Run H1 diagnostic** (cheap):
   ```bash
   cd /home/jasper/Repositories/MasterThesisCode
   # Re-prepare h=0.73 with a fresh seed
   uv run python scripts/prepare_detections.py \
       --workdir simulations/closure_h0p73_h1diag \
       --input-crb simulations/cluster_run_phase46_merged_20260504/cramer_rao_bounds.csv \
       --seed 211 --force
   # ... need to set up CRB rescaling first (test_23 to make rescaled CRB at h=0.73)
   ```
   Alternative: lift the seed-bootstrap protocol from
   `scripts/bias_investigation/test_22_dh_double_count.py` and adapt.

2. **Tabulate σ_realization vs σ_boot at h=0.73 in 2D.** If σ_realization
   ≈ 0.02 (matching the existing seed-dependence finding), z_panel drops
   from 65 to ~3 — and that's the actual number to report. H1 confirmed.

3. **If H1 doesn't fully account for the 2D extremes** (e.g. the +55σ at
   h=0.60), pursue H2: read `posterior_combination.py` and trace the 2D
   path.

4. **Document a new section §3.14 in `docs/H0_BIAS_RESOLUTION.md`** with
   findings + fix.

---

## Files to read first

| Order | Path | Why |
|---|---|---|
| 1 | `docs/H0_BIAS_RESOLUTION.md` §1, §2 (Multi-truth panel partial), §3.13 | What's been fixed, what's the new finding |
| 2 | This file | Hypotheses and starting points |
| 3 | `scripts/bias_investigation/outputs/phase45/multi_truth_sweep_partial4truths_20260505_115451.json` | Ground-truth numbers |
| 4 | `scripts/bias_investigation/test_24_multi_truth_bias_sweep.py` (esp. `analyze_one_truth`) | How σ_boot is computed |
| 5 | `master_thesis_code/bayesian_inference/posterior_combination.py` (`combine_log_space`) | Where the joint posterior is assembled |
| 6 | `master_thesis_code/bayesian_inference/bayesian_statistics.py` (`single_host_likelihood`, both channels) | Per-event likelihood for 1D vs 2D |
| 7 | Memory: `finding_seed_dependent_map.md` | The σ_boot blindspot we already documented |
| 8 | Memory: `finding_migration_footgun.md` | Side-quest from yesterday — not load-bearing here |

## Files NOT to touch (panel running)

- `scripts/bias_investigation/run_multi_truth_sweep.sh`
- `cluster/evaluate_closure_h_true_finegrid.sbatch`
- Any local `simulations/closure_h*/` workdirs
- `simulations/cluster_run_closure_h*_finegrid/posteriors{,_with_bh_mass}/`
  (these get refreshed by the orchestrator as truths complete)

It's safe to read these dirs but don't write into them, and don't submit
sbatch jobs to dev_cpu_il while the orchestrator is using QOS=dev. If you
need cluster compute for a diagnostic, queue it on `cpu` or `cpu_il` (high
walltime estimate but won't fight QOS=dev).

---

## Cluster status (snapshot at handoff time)

- **Panel job in flight:** `4227491_*` on `dev_cpu_il` (h=0.75)
- **Pending after that:** h=0.80, h=0.85
- **Seed=300 GPU sim:** cancelled last night (`4216323`); 41/50 tasks
  completed → 1125 SNR≥20 events ⊕ 424 from Phase 45 = 1549 in
  `simulations/cluster_run_phase46_merged_20260504/cramer_rao_bounds.csv`.
- **SSH master alive:** PID 851876, ControlPersist 10h. Re-establishes on
  any `ssh bwunicluster` if killed.

---

## What success looks like

A new entry in `docs/H0_BIAS_RESOLUTION.md` §3.14 (or §4.3) that:
1. Explains why the 2D z=65 collapsed (or fixes the underlying mechanism)
2. Reports σ_realization at h=0.73 in 2D and the resulting z
3. States whether the panel verdict is now PASS, MARGINAL, or still FAIL
   under the corrected uncertainty

If H1 is the answer (σ_boot was just blind), the deliverable is a
methodology change in `test_24` to use realization-bootstrap (or to caveat
the σ_boot tightly). If H2/H3 is the answer, the deliverable is a code fix
in `posterior_combination.py` or `bayesian_statistics.py` plus a re-run of
the 2D analysis on at least h=0.73 to confirm the bias collapses.

---

## Status of TODO list at handoff

| Done | Pending |
|---|---|
| Phase 45 + seed=300 merge → phase46-merged | Read JSON, interpret 4 panel verdicts (partial done) |
| Build phase46-merged CRB | Commit panel results (full panel) |
| Update DATA_INVENTORY.md | This 2D investigation |
| Launch full 7-truth panel | |
| Refresh interactive figures (commit `8e1d4a4`) | |
| Refresh thesis + paper PDFs (commit `39e2ae6`) | |

---

## One more thing

When the panel finishes (~15:00) the orchestrator's auto `test_24` run will
overwrite `multi_truth_sweep.json` with the full 7-truth verdict. The
partial snapshot is preserved at `multi_truth_sweep_partial4truths_20260505_115451.{json,png}`
so don't worry about losing it. Compare full vs partial when full lands —
the 3 missing truths (0.75, 0.80, 0.85) might pull the 2D z down (negative
biases) or push it further up. Use that signal alongside H1–H4.

Good hunting.
