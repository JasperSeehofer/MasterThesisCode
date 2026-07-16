# Next-session kickoff prompt (2026-07-11 → next)

Paste the block below as the first message of the next session.

---

Continue the deep-incompleteness bias investigation on branch
`physics/zero-host-completion-fallback` (all of today's work is committed +
pushed, tip `b89d3b7`). Start by reading `.planning/BIAS-INVESTIGATION-20260710.md`
ledger item **[L7]** and the four `results/pp_coverage_*_20260711/SUMMARY.md`
files — that is the current state. Short version: the deep-incompleteness HIGH
bias is decomposed = dominant membership-support **kernel leak** (removed by the
new `--mixture-mode exact`) + a small σ_z-independent **floor** (+0.002…+0.005 in
h) that is NOT the prior (N-3) and NOT the p_det-inside factor (27m, refuted).

**Model / cost discipline (applies all session):** default to Sonnet. If you run
a Workflow, set the agent `model` to `sonnet` for ordinary find/verify/sweep
stages and only escalate to a stronger tier for a genuinely hard synthesis or
adjudication stage. When you spawn subagents directly (Agent tool / GSD
executor+planner), pick Sonnet unless the task is clearly reasoning-bound. Do NOT
launch a Workflow at all unless the task actually needs multi-agent fan-out —
these floor/N-4 probes are single-threaded harness runs, so plain `/gsd:quick`
(planner+executor) or even inline execution is the right tool. Keep it lean.

**First action — debrief the flagged items** (I deferred these to keep the last
session lean; do them before new probes): run `/scribe-debrief` in THIS session.
Two reusable lessons to file:
1. **Pre-registration discipline caught a partially-wrong hypothesis** — writing
   the CALIBRATED/BIASED prediction into the RUNBOOK *before* running (tasks
   117/27m) turned "gray mixture is the escape hatch" and "p_det-inside is the
   floor" into clean, falsifiable, and falsified results instead of motivated
   readings.
2. **Coarse MAP-grid quantization masks small ensemble shifts** — the default
   h-grid (step 0.004) quantizes sub-grid MAP-mean shifts to exact ties on tiny
   test configs; recurred twice. Fix: assert on the continuous per-branch tilt
   diagnostics (`dlogL_dh_{host,completion}_mean`) or drop to `h_step=0.001`.
   The ensemble mean stays unbiased; only strict-ordering tests need the finer
   grid.

**Then, the ranked next probes (all local, harness-only, no /physics-change —
pp_coverage stays production-independent):**

- **N-floor (⭐ finish the decomposition): the σ(dL_obs)-vs-σ(dL_true) noise-model
  candidate.** The floor is σ_z-independent, prior-insensitive, grid-robust, and
  O(σ_f²) in scale — every property points at the inference GW-likelihood using a
  constant σ = σ_f·dL_obs while the generative noise is σ_f·dL_true (z-dependent
  inside the integral, with the 1/σ(z) normalization variation). Probe: evaluate
  the inference σ INSIDE the z-quadrature (σ_f·A(z)/h, include the 1/σ(z)
  prefactor), run a 2×2 with `--pdet-in-numerator` at the deep cells, and add a
  cheap n_events scaling check (does the floor behave like a skewed-MAP-statistic
  artifact, given calibrated controls carry −0.002…−0.003 MAP offsets of the same
  size and cov68 is largely in-band?). If the z-dependent-σ variant flattens the
  floor ⇒ decomposition complete, floor = harness noise-model approximation, not a
  production concern. Pre-register the prediction in the RUNBOOK first.

- **N-4 shallow +0.0138 (the OTHER open regime — seed600 frozen venue, comp_frac
  0.4%, L-A mechanism ~zero here so it is genuinely separate):**
  (a) re-parameterize the harness to the seed600 regime — detected z_median 0.046
  (needs D50/W_PDET knobs so the venue sits at D50 ≈ 0.2–0.3 Gpc, not the default
  1.85), venue-matched σ_z — and ask whether a *calibrated* estimator shows a
  +0.013-like offset in THAT regime;
  (b) jackknife / influence analysis on the EXISTING seed600 per-event likelihood
  JSONs (`results/pv_correction_test_20260703/run_live` and
  `results/seed600_ab_20260710`, on disk — no re-eval): is +0.0138 driven by a
  small heavy-tailed subset or spread evenly? Beyond these two, systematic-vs-
  scatter needs the multi-seed campaign — do not force it locally.

- **N-5 (optional, 2D):** re-run the G7row9 494-event driver at `fc45d1f` to see
  whether the 0.7697(7-pt)/0.787(17-pt) subsample spread collapses under the
  `713fbd1` D_g fix (full-venue is already 0.7546).

**Do NOT re-attempt** (adjudicated this week, ledger [L7] + anti-repetition ledger
in `.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`): gray mixture (amplifies),
conditioned inverse (doesn't rescue), prior tilt (negligible), p_det-inside
(refuted), and all previously-exonerated suspects (Fisher frame, catalog
Jacobian, Ω_m era term, D(h) structure).

**Still user-gated (do not decide autonomously):** D1 (depth framing — now has
strong evidence: exact mode calibrates coverage, prior sensitivity negligible, so
deep incompleteness is NOT intrinsically un-calibratable; truncation stays a
robustness bound), D2 (PR merge order #22 → #31 → #32), D3 (Paper A venue caveat),
D4 (2D residual +0.025), D5 (time allocation). And the production-side soft
f(z)-weighted-kernel correction candidate from N-2d is /physics-change +
literature (Gray 2020, CFH 2018, ICAROGW) + user approval BEFORE any production
code — flag it, don't start it.

**Cluster is still down** (security incident, est. return early next week). When it
returns, the runbook is unchanged (`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md`
L-F): security hygiene → preflight READY → rsync depth15 pool → h=0.705 re-run →
deploy merged branch per D2 → EXP-40 (watch: interior-but-biased-HIGH, and the
post-#29 mixture may overshoot MORE than two-branch) → only then seeds 2000–6000.

---
