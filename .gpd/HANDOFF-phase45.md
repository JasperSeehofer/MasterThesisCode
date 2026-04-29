# Phase 45 Handoff — Residual MAP Bias After Phase 44

**Status:** ready to start. Phase 44 shipped; cluster confirmed MAP shifted 0.860 → 0.7650 on 412 events. Truth h=0.73, residual +0.0350 (+5%). This handoff captures everything Phase 45 needs to start without re-loading Phase 44 context.

---

## Phase 44 result (one-paragraph summary)

`detection_probability_without_bh_mass_interpolated_zero_fill` had a left-side cutoff at `dl_centers[0] = dl_max(h)/120` that scaled as 1/h. Below that threshold, `p_det = 0` was returned. Phase 44 removed the cutoff (commit `3697bdd`); the interpolator's existing `fill_value=None` (nearest-neighbour) now returns the genuine first-bin estimate `p̂(c_0) ≈ 0.55` for `d_L < c_0`. Cluster re-eval (jobs 4160638/4160639) on production seed200 (412 events post-SNR-filter): **MAP = 0.7650** vs pre-fix MAP = 0.860. All 4 zero-handling strategies now produce identical MAP — confirming no events are being suppressed by zero-handling logic. 68% equal-tailed interval [0.750, 0.765].

The +145.7 log-unit pathology is gone. The +0.035 residual is a different (smaller, predictable) effect.

## Headline numbers (do not reload these)

| Quantity | Value |
|---|---|
| Truth (injection h) | 0.7300 |
| Pre-fix MAP | 0.860 |
| Post-fix MAP | **0.7650** |
| Residual bias | +0.0350 (+5% above truth) |
| 68% equal-tailed interval | [0.750, 0.765] (does NOT contain 0.73) |
| Median | 0.7600 |
| n_events post-SNR-filter | 412 |
| dl_centers[0] at h ∈ {0.60, 0.73, 0.86} | {0.1214, 0.0998, 0.0847} Gpc |
| n_total[0] (first bin injections) | 312 (constant across h) |
| p̂(c_0) at h ∈ {0.60, 0.73, 0.86} | {0.471, 0.545, 0.596} |
| Cluster wallclock per task | ~2:30 on `cpu` partition |
| Total cluster runtime (38-task array + combine) | < 5 min once running |

## Hypothesis chain

Three competing/composite explanations for the +0.035 residual:

### H1 — First-bin underestimate (Phase 44 plan §8 hypothesis)

`p̂(c_0) ≈ 0.55` is the histogram-derived first-bin mean p_det. Below `c_0`, nearest-neighbour returns `p̂(c_0)`. But the *true* `p_det → 1` as `d_L → 0` (any source close enough is detectable). So we're systematically *underestimating* `p_det` for `d_L < c_0`, which underestimates L_comp at low h (where `c_0` is largest). This biases MAP toward higher h.

Plan §8 recommended remedy: **Alternative C — explicit `(d_L=0, p_det=interp(c_0))` anchor** in the interpolator. But that anchor is wrong as written — anchoring at `interp(c_0)` reproduces nearest-neighbour. The *correct* Alternative C should anchor at `(d_L=0, p_det=p_max)` where `p_max ≤ 1` is the true asymptote (and 1 may be too high — see "open questions" below).

### H2 — Statistical fluctuation

412 events isn't enormous. 68% interval width is 0.015. Residual is ~2-3σ. Could partly be sampling noise rather than systematic. **Bootstrap is the only way to know.**

### H3 — Other physics caveats (not yet checked for cumulative effect)

CLAUDE.md lists 4 MEDIUM-priority physics bugs:
- `physical_relations.py:72` wCDM params silently ignored
- `bayesian_inference.py` hardcoded 10% σ(d_L) (not the per-source CRB)
- `constants.py:29-30` WMAP-era cosmology (Ω_m=0.25, H=0.73 vs Planck 2018)
- `datamodels/galaxy.py:64` redshift uncertainty `0.013(1+z)^3` non-standard scaling

Individually, none should drive +0.035. Cumulatively, plausible but unlikely.

## Suggested investigation order

**Step 0 — Bootstrap to quantify statistical vs systematic.** This MUST come first.

Resample 412 events with replacement N=1000 times; recompute MAP per resample (using the existing 38-h posterior cache, no re-eval needed); report the distribution of MAP. Outcomes:

- MAP = 0.7650 ± 0.02 (1σ) overlapping 0.73 → residual is statistical; **accept and document, no further fix**
- MAP = 0.7650 ± 0.005, doesn't overlap 0.73 → systematic; **proceed to Step 1**
- Anything else → re-think

This is fast (no cluster cycles) — the per-h posteriors at `results/phase44_posteriors/h_*.json` are already cached.

**Step 1 — Investigate first-bin density.**

Histogram d_L distribution within `[0, 2*c_0(h=0.73)]` (i.e., the first injection bin) for the actual injection campaign data. Is it uniform? Skewed toward the upper edge? Toward the lower edge? Two cases:

- **Skewed toward upper edge (high d_L within the bin)** — `p̂(c_0)` mean is dominated by the high-d_L (low-p_det) end; we underestimate p_det near d_L=0. Confirms H1.
- **Roughly uniform** — `p̂(c_0)` is the unbiased mean; H1 is wrong; reach for H3.

Reuse `analysis/grid_quality.py` (already exists) — it has Wilson-CI infrastructure and per-bin counts.

**Step 2 — Conditional fix (depends on Step 0/1):**

| Step 0 outcome | Step 1 outcome | Action |
|---|---|---|
| Statistical | n/a | Document MAP=0.7650 ± σ as expected; no fix |
| Systematic | First-bin upper-edge skew | Implement Alternative C with appropriate p_max anchor |
| Systematic | First-bin uniform | Open new debug session — H1 is wrong, look at H3 |
| Systematic | First-bin lower-edge skew | `p̂(c_0)` overestimates (events near 0 are easier to detect, dominate bin avg). Phase 45 should be: anchor at *lower* p_max, not higher. |

**Step 3 — Cluster verification of the chosen fix (if any).**

Submit a fresh eval+combine on production seed200 using `cluster/submit_phase44_eval.sh` as a template. Acceptance: MAP ∈ [0.72, 0.74] AND 68% interval contains 0.73.

## Critical files

| File | Why |
|---|---|
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:478–502` | `_build_grid_1d` — where Alternative C anchor would land if applied |
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:672–717` | The Phase 44 fixed `_zero_fill` function; do not regress |
| `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py::TestZeroFillBoundaryConvention` | 4 regression tests guarding Phase 44 — must continue passing |
| `cluster/submit_phase44_eval.sh` | Template for cluster re-eval. Note: queue dynamics — `dev_cpu_il` rejects 38-task arrays (QOS limit); use `cpu` partition with `--partition=cpu` override. |
| `results/phase44_posteriors/` | Current MAP=0.7650 baseline; per-h JSONs feed bootstrap directly |
| `analysis/grid_quality.py` | Existing Wilson-CI + per-bin diagnostics infrastructure |
| `scripts/bias_investigation/` | 7 existing diagnostic scripts (d_L distribution, etc.) — reuse |

## Open questions for Phase 45 to clarify before fixing

1. **What's the true p_det asymptote at d_L → 0 for LISA EMRIs?** Probably not 1.0 — sky position, inclination, and intrinsic loudness all matter even at zero distance. A conservative estimate from the actual injection campaign: subset injections with `d_L < 0.02 Gpc` (well into the first bin's interior) and compute their detection fraction. This gives the empirical asymptote.
2. **Is `c_0(h)`-dependent fix safe?** The current fix removed an h-dependent threshold; we don't want to reintroduce one. Any `(d_L=0, p_det=A)` anchor that's h-independent is safe. An h-dependent `A` would be a regression.
3. **Does the 4σ window for L_comp actually reach `d_L = 0`?** For the bias-driving close events with `d_L ≈ 0.05–0.10 Gpc` and `σ(d_L) ≈ 0.005`, the window is `[0.04, 0.10]` — well above 0. The Alternative-C anchor at `d_L = 0` only matters for an integrand evaluated near 0, which only happens for very-close events with very-large σ. **Check whether ANY event in the 412-event production set has its 4σ window touching `d_L = 0`** — if not, Alternative C will have zero effect on the production posterior and Phase 45 needs a different angle.

## What's out of scope for Phase 45

- The 4 MEDIUM-priority physics caveats (separate Phase 46+ if Step 1 implicates them)
- Sky-frame / coordinate work (closed by Phase 43)
- Any change to L_cat — only L_comp / D(h) / p_det stack
- Re-running the simulation campaign (CRBs are fine; only re-evaluation is needed)

## Suggested kickoff

```bash
# In a fresh session, in /home/jasper/Repositories/MasterThesisCode:
# 1. Read this handoff (~3k tokens)
# 2. Read the resolved Phase 44 debug for full mechanism context:
cat .gpd/debug/resolved/map-0p86-lcat-explosion.md | head -200
# 3. Spot-check current state:
git log --oneline -5
ls results/phase44_posteriors/ | head
# 4. Start with Step 0 (bootstrap) — write a small inline diagnostic, no need for /gpd:plan-phase yet.
```

When Step 0 + Step 1 results are in, *then* invoke `/gpd:plan-phase 45` with the diagnosis-locked-in. Plan-first-then-investigate would waste cycles if H1 turns out to be wrong.

## Repository state at handoff

- Branch: `main` (synced with `origin/main`)
- Latest commit: `807d6c0` (Phase 44 wrap-up)
- CPU tests: 557/557 passing
- Cluster: nothing pending; results landed at `results/phase44_posteriors/`
- Resolved debug session: `.gpd/debug/resolved/map-0p86-lcat-explosion.md` (full mechanism + falsifiability evidence)
- Vault debrief: filed at `vault/wiki/log.md` 2026-04-29; 3 new tentative cross-project patterns awaiting caretaker verification
