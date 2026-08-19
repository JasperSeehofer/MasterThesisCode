# Runbook — next session (written 2026-08-19, supersedes RUNBOOK_NEXT_SESSION_20)

**Read first:** `results/prod2d_closure_20260818/PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md`
(the registration of record incl. Part-VII amendments and both fill-ins) +
`VERIFIER_PRECHECK_PROD2D.md` + ledger rows #125–#127. Then this file top to bottom.

## 0. FIRST ACTIONS — set up the watchers (the old session's monitors are DEAD)

Cluster job **6364821** (prod2d closure+landscape, 18 cells, 14 h walltime, submitted
2026-08-19 ~01:30, expected terminal ~12:00–15:30 at the latest) may still be running or
already terminal. Immediately:

1. `timeout 90 ssh -o BatchMode=yes bwunicluster 'sacct -j 6364821 --format=State,Elapsed -n -X | head -1'`
2. If RUNNING: arm a persistent Monitor polling that sacct line every ~600 s, emitting ONLY
   on state change (dedupe on the State field alone, not Elapsed — the Elapsed-included
   version spams every poll), terminal on
   COMPLETED/FAILED/CANCELLED/TIMEOUT/OUT_OF_MEMORY/NODE_FAIL.
3. If TERMINAL (any kind): retrieve everything that exists —
   `rsync -az "bwunicluster:$(ssh bwunicluster 'ws_find emri')/run_prod2d_20260818/*.json" results/prod2d_closure_20260818/cells/`
   Cells are per-cell JSONs, idempotent: on TIMEOUT/partial, resubmit completes only the
   missing ones: `ssh bwunicluster 'cd ~/darksiren-emri && WS=$(ws_find emri) && sbatch --export=ALL,RUN_DIR=$WS/run_prod2d_20260818 results/prod2d_closure_20260818/run_prod2d.sbatch'`
   (**RUN_DIR export is mandatory** — the first submission 6364803 failed in 20 s without it.)
4. Background-shell gotchas of record: plain `Bash run_in_background` jobs got reaped twice
   → for anything long-running local use `nohup setsid … & disown` + a Monitor on the log;
   ssh/git to the cluster is SLOW (pushes ~5–7 min; run in background, NEVER chain a
   branch-delete after a merge in one command — that combination destroyed a pushed branch
   once this session).

## 1. State at reset (2026-08-19 ~07:00)

- **Front of record: production-2D closure + catalog-quality landscape** (row #127; author
  directive verbatim there — closure, cluster, overnight, autonomous; branch calls return
  as [RULE]s). Registration frozen at `d6fc1ccf`; verifier Part VII (P7-1…P7-8) applied
  verbatim; arm-validity preflights READY; cluster repo at tag `prod2d-closure-base`.
- **T0 (production-native) is BANKED** (`tier0_output.json`, registered run, N-0 gate PASS):
  σ_boot(2D) = 0.0114 (iiib) / 0.0121 (joint_r1) ⇒ **z = 4.75 / 5.53 — "event-draw scatter
  alone cannot own the offset"**; jackknife-889 **ROBUST** (iiib Δ +0.054 → +0.065 WITHOUT
  event 889 — it pulls toward truth; offset is a broad ensemble effect in both venues);
  production-native "1D starves" legs fire in both venues (H-L1-prod); truncation
  diagnostic clean. Production offsets of record (trapezoid convention): **+0.054 (iiib) /
  +0.067 (joint_r1)**, single realization seed61000 shared by both venues (P7-8: never two
  independent confirmations).
- **5/18 cluster cells were already done + retrieved at reset** (in `cells/`), INTERIM:
  - **(σ_z=0.035, σ_m=0.55) off:** V-deep 2D bias **+0.038…+0.042, cov68 0.00**, map_std
    0.009–0.011 — the production-mapped σ_M puts the harness 2D displacement in
    production's magnitude class (4× the σ_m=0.30 class, matching H-T1b's ×2–4).
  - **σ_z ladder at σ_m=0.55 (off):** 2D bias +0.040 → +0.007 (σ_z 0.010) → +0.002
    (σ_z 0.002): the 2D mass-error bias is **photo-z-mediated — a σ_M × σ_z coupling**.
  - 1D off-basis: calibrated at 0.035 (−0.002, cov 0.63–0.76); at 0.010/0.002 a small
    **−0.003…−0.004 1D floor** persists and becomes coverage-breaking as widths shrink
    (partly quantization-flagged; N-c certified ≤0.0005 at probe scale — tension, logged,
    NOT yet interpreted).
  - V-prod off: the known raised-d50 venue bias (+0.008…+0.015 both channels, descriptive).
- **Rows #125/#126 (previous front):** G-1/G-2 ratified; [P3] presentation pick = **(b)
  resolved-in-paper**; paper work granted, sequenced after this closure.
- **GitHub push REJECTED** (pre-receive hook, reason not yet captured — housekeeping item;
  run `git push origin main` once interactively or capture the full remote: lines; local +
  cluster copies are the records). Local repo is ahead of origin by ~13 commits.

## 2. When the job is terminal — the readout pipeline (all registered)

1. Retrieve cells (item 0.3). Copy the job log:
   `scp bwunicluster:~/darksiren-emri/prod2d_closure_6364821.out results/prod2d_closure_20260818/`
2. `cd results/prod2d_closure_20260818 && uv run python readout_prod2d.py --registered cells/`
   → `readout_prod2d_output.json` (scores H-T1a with its engagement precondition, H-T1b,
   H-L1-harness, H-L2, N-1 continuity, pairs, rail/quantization gates).
3. Assemble the **closure budget** exactly per prereg §4 (P7-4 arithmetic:
   r_v = Δ_v − s_Edd(−0.020); σ_total = σ_boot ⊕ u(s_Edd); **only production-native
   magnitudes**; harness legs = class/sign support only; jackknife modifies which Δ is
   quoted, not σ_total) → branch B-OWNED-SCATTER / B-OWNED-BUDGET / B-UNOWNED per venue.
4. Landscape tables (bias/σ_real/cov68/RMS per grid rung, both channels, off-basis 1D
   column; quantization upper-bounds at good rungs) + mission overlay (presentation-only:
   GLADE 0.035 / LSST-class 0.02–0.03 / spec-z 0.002; σ_m: R&V-current 0.55-frac /
   improved-EM ~0.25 / optimistic 0.02).
5. A7 comprehension-first campaign report + append VERDICT to the prereg + ledger row +
   decisions table to the author. Cluster fill-in append: per-cell timings, any resubmit.
6. Chronicle: /chronicler if reportable signals (there are several — see §5).

## 3. The remaining-residual plan (start while cells run; prereg-first for anything new)

The residual after the budget will likely be Δ_v − (event-scatter allowance) − (documented
−0.020) with the σ_M × σ_z coupling as the candidate CLASS owner (interim harness evidence
is strong: right sign, right magnitude class at production-mapped errors, collapses with
either σ). The approach, in order of cost:

1. **Production-native class test (FREE, decisive — do this first).** Extend the T0
   machinery: regress the per-event 2D h-slopes (already computable from
   `event_likelihoods.csv`) against per-event observables from
   `prepared_cramer_rao_bounds.csv` + handler outputs: per-event σ_M (R&V15 propagated),
   z/photo-z context, in_catalog, SNR, completion share (`g_frac`, `share_cat`). The
   coupling class predicts the positive slope mass concentrates in events where the mass
   factor acts against a photo-z-smeared support. If the production per-event structure
   matches the class prediction → the residual is CLASS-OWNED production-natively (not just
   harness-class), closing the P7-4 transfer gap. Pre-register the regression + bands
   BEFORE running (cheap prereg, verifier one-item; the discipline holds even for free
   reads — T0's value came from exactly that).
2. **Mechanism derivation (top-tier, no compute).** Derive WHY σ_M × σ_z produces a
   positive h-pull in the 2D completion/catalogue geometry (candidate: over-tight effective
   mass kernel at smeared z ⇒ Malmquist-type asymmetry in the (d_L, M_z) overlap ⇒ h-tilt).
   Deliverable: predicted sign + scaling (∝ σ_M²·f(σ_z)?), checked against BOTH the harness
   ladder (T2 grid gives the response surface for free) and the production regression of
   item 1. This is the same playbook that closed the 1D convention story (mechanism → 3%
   agreement → repair).
3. **The fix fork (author-gated, /physics-change if code):** three registered options to
   present WITH the mechanism in hand, none pre-decided:
   (a) **model correction** — if the derivation shows the estimator's 2D kernel under-uses
   the known σ_M (e.g. the Eddington-in-M treatment incomplete at the completion leg), the
   correct-form kernel is a physics-change proposal with the T2 response surface as its
   verification bed;
   (b) **importance/marginalization repair** — marginalize the mass-observation error where
   it is currently point-estimated (same gate);
   (c) **document-as-systematic** — quote the 2D constraint with the coupling-class
   systematic and the landscape as the mitigation map (no code change; consistent with
   correctness-over-bias-removal if (a)/(b) are found not to be defects but information
   limits).
4. **The −0.004 1D spec-z floor** (small, separate): one decider cell class — n_z_quad
   doubling at FULL R at the spec-z rung (the probe-scale N-c may have been
   quantization-blind) vs a truth-grid-offset probe (is it grid-alignment?). ~2 CPU-h
   local. Only after the main readout; keep it out of the closure's critical path.
5. **Sequencing:** item 1 can run TODAY while cells finish (production-native, free,
   independent). Item 2 in parallel (derivation). Budget assembly needs the fused cells.
   The fix fork returns to the author with mechanism + budget together.

## 4. Standing constraints

Autonomy grant of row #127 covers THIS closure front (execution through readout; branch
calls presented, not adjudicated). Prereg-first + verifier pre-check for every new
measurement incl. free reads. Top-tier cap; venue-scoping binds both ways (P7-4's
production-native-magnitudes rule is the closure's spine — do not let harness magnitudes
leak into the budget). Append-only ledgers/preregs. Workspace expires 2026-09-23
(0 extensions) — copy finals off the cluster workspace when the campaign closes. Author WIP
in book/design/* + gen_ch00.py uncommitted — do not touch. GitHub push rejection: capture
reason, resolve as housekeeping (may be a hook on large files or protected branch — do NOT
force-push).

## 5. Chronicle candidates (for /chronicler at natural end)

RUN_DIR-export sbatch failure mode; monitor-dedupe-on-state-only pattern; the
push→merge→branch-delete-chained-command incident; untracked-collision merge resolution
with md5 verification; the σ_M × σ_z coupling discovery arc (interim); T0's
bootstrap-vs-jackknife inversion (event 889 helps, not hurts).

## 6. Resume recipe (one line)

Watchers (§0) → job state → if terminal: §2 pipeline → §3 items 1–2 in parallel → report +
decisions to author. If still running: §3 item 1 (free production-native regression prereg)
first.
