# Next-session kickoff prompt (2026-07-12 → next)

Paste the block below as the first message of the next session.

---

Continue the H₀ bias investigation on branch `physics/zero-host-completion-fallback`
(all work committed + pushed, tip `038bf82`). Start by reading `.planning/STATE.md`
(Quick Tasks table, top rows) + `.planning/BIAS-INVESTIGATION-20260710.md` ledger
items **[L7]** (deep floor) and **[L8]** (shallow venue), plus the two newest
SUMMARYs: `results/pp_coverage_noisemodel_20260711/SUMMARY.md` and
`results/pp_coverage_shallowvenue_20260711/SUMMARY.md`. That is the current state.

**Short version — the bias story is now mechanistically CLOSED at the harness level:**
- **Deep-incompleteness bias FULLY DECOMPOSED** = dominant membership-support **kernel
  leak** (removed by `--mixture-mode exact`, 260711-117) + a σ_z-independent
  **noise-model floor** (the joint σ(dL_obs)-vs-σ(dL_true) width mismatch + p_det-inside,
  removed ~85–90% by `--sigma-model-in-likelihood --pdet-in-numerator`, 260711-hx1). The
  const-σ floor is a *real asymptotic bias* (flat in n, cov68 collapses); tiny 2nd-order
  residual ≈15× below campaign σ_boot.
- **Separate shallow +0.0132 (seed600, comp_frac 0.4%) EXPLAINED** = estimator-intrinsic
  **σ_z/z-at-low-z truncated-volume-kernel Eddington effect** (260711-iic): the calibrated
  volume kernel reaches +0.030 at z_med 0.044 but only at σ_z=0.035 (vanishes at σ_z≤0.015);
  seed600 jackknife confirms the residual is broad/systematic, not outlier-driven.
- **Both regimes converge on ONE production fix** (see user-gated, below).

**Model / cost discipline (all session):** default to Sonnet. Do NOT launch a Workflow
unless the task genuinely needs multi-agent fan-out — the remaining probes are
single-threaded harness runs, so `/gsd:quick` or inline execution is right. When you do
spawn subagents (Agent tool / GSD executor+planner), pick Sonnet unless the task is
clearly reasoning-bound. Keep it lean. Pre-register CALIBRATED/BIASED predictions in the
RUNBOOK before any run, and assert on the continuous tilt diagnostics or a fine h-grid,
not the coarse MAP grid (two lessons filed to the vault this week).

**Remaining LOCAL work (harness-only, no /physics-change):**

- **N-5 (optional, 2D channel — the last local item):** re-run the G7row9 494-event
  driver at `fc45d1f` and check whether the 0.7697(7-pt) / 0.787(17-pt) subsample spread
  collapses under the `713fbd1` D_g fix (the full-venue number is already 0.7546). Fresh
  context helps — this is a different channel from the 1D work above.

- **Load-bearing input that CLOSES the N-4 shallow attribution (cheap, no re-eval):** what
  is seed600's *effective redshift-uncertainty at z ≈ 0.046*? The [L8] Eddington mechanism
  needs σ_z/z ~ O(1); if seed600 uses that (photo-z-like), the +0.0132 is (partly) this
  effect; if it is small spec-z (σ_z/z ≪ 1), the shallow residual is something else. Check
  the seed600 catalogue / CRB redshift-error model (e.g. `results/pv_correction_test_20260703/`
  metadata, the reduced GLADE catalogue z-error column). This is the single fact that turns
  N-4's "reproduced the mechanism" into "attributed to seed600."

**Do NOT re-attempt / re-open** (adjudicated — ledger [L7]/[L8] + anti-repetition ledger in
`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`): gray mixture (amplifies), conditioned
inverse (doesn't rescue), prior tilt (negligible), p_det-inside ALONE (refuted), σ-model
ALONE (over-corrects), the deep floor (CLOSED), the shallow regime mechanism (CLOSED), and
all previously-exonerated suspects (Fisher frame, catalog Jacobian, Ω_m era term, D(h)
structure).

**User-gated — do NOT decide or start autonomously:**
- **The production kernel correction** — now BOTH regimes point to the same change: a
  **z≥0-truncation-aware / photo-z-marginalized volume host-z kernel** fixes the deep
  membership-support leak AND the shallow σ_z/z Eddington effect in one move. This is
  `/physics-change` + literature (Gray 2020; Chen–Fishbach–Holz 2018; Mastrogiovanni/ICAROGW;
  the commission-d2 volume/Eddington correction) + user approval BEFORE any production code.
  Flag it, don't start it.
- D1 (depth framing — strong evidence now: deep incompleteness is NOT intrinsically
  un-calibratable at the estimator level; truncation stays a robustness bound), D2 (PR merge
  order #22 → #31 → #32), D3 (Paper A venue caveat), D4 (2D residual +0.025 → N-5), D5 (time
  allocation).

**Cluster** (est. return early this week — verify): runbook unchanged
(`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md` L-F): security hygiene → preflight READY →
rsync depth15 pool → h=0.705 re-run → deploy merged branch per D2 → EXP-40 (watch:
interior-but-biased-HIGH; post-#29 mixture may overshoot MORE than two-branch, and it carries
BOTH the leak and the floor same-signed HIGH per [L7]) → only then seeds 2000–6000.

---
