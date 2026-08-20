# Runbook — next session (written 2026-08-20, supersedes RUNBOOK_NEXT_SESSION_24)

**Read first:** ledger rows **#145 → #148** (plus the two addenda to #145), then
`results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md` **v2** — that document is the
work. Runbook 24 covers the sentinel arc and stays valid as background; rows #127–#144 and
`docs/RETROSPECTIVE_D1_20260820.md` are the deeper background. **Do not redo any of it.**

## 0. Where the campaign stands

The 2026-08-20 session cleared everything between the campaign and its settling measurement:

- **Rows #145/#146** — the mirror's log-space `-1e300` sentinel manufactured `mean_h` = the
  `H_GRID_41` midpoint ≡ `H_TRUE` in 25/123 seeds. Both fixes implemented; legacy paths retained and
  proven to reproduce the banked fleet **123/123** bit-exactly. **A15 ADOPTED.** Fully-corrected
  numbers of record: b0 +0.0296 · bsig005 +0.0362 · eden05 +0.0139 · eden2 +0.0321 · bf1 **+0.0358**
  (the old positive control **fails**) · bout −0.1287 · **bsel −0.1083** · bself −0.1126 · bden −0.1159.
- **Row #147** — the `g_frac`-NaN generator question: **CONTROL-SAFE**, closed. Those events are
  h-inert to `0.000e+00` exactly, and B-SEL carries 0 in 12 seeds.
- **Row #148** — the C-SG positive control is **REGISTERED (v2)**, not implemented, not run.

## 1. THE NEXT ACTION IS FREE — do it before spending any CPU

`PREREGISTRATION_SELFGEN_CONTROL.md` §9 (pre-check O2). The 12 banked B-SEL diagnostics under
`results/prod2d_closure_20260818/arm_event_likelihoods/bsel_seed*/` carry `alpha_G_phi`,
`L_cat_no_bh`, `B_num`, `D_tilde_phi` per event per h.

**Recompute the posterior with `L_cat_no_bh ≡ 0`** — the pure-completion arm — at zero compute.
Motivation: **128/174 (73.6%)** of B-SEL events have an active *impostor* catalogue leg, with
impostor share of the per-event numerator reaching 0.647 at the 99th percentile and 0.821 at max.

> **If the impostor leg carries part of the −0.1083, C-SG's design must change before it runs.**
> If it does not, §5's generator–model mismatch is quantified rather than asserted.

Then: **the mandatory 4-seed C-SG-F pilot** (§6) whose only outputs are `σ̂_seed` and per-seed
`σ_h`. **No band in the registration is valid until that pilot reports** — they were all deleted for
exactly that reason (§0 item 6 of row #148).

## 2. Why v2 exists — the five things that were wrong with v1

Its own adversarial pre-check returned **NOT-READY**, 11 required amendments. Every decisive finding
was re-derived before adoption; two refuted design choices already written down:

1. **The measurement kernel direction.** `N(d_L/d̂; 1, σ_dL/d̂) ≡ d̂·N(d̂; d_L, σ_dL)`, verified to
   1e-12. The estimator's `ratio` kernel **is** the fixed-σ linear Gaussian, so **B-SEL's linear
   draw was matched**, and v1's ratio draw would have injected the `d_L ∝ 1/h` factor — the
   campaign's own predicted defect — into the generator.
2. **Selection applied twice.** `S̄_φ` **is** the marginal detection probability
   (`bayesian_statistics.py:1932-1975`), so drawing with it *and* accepting with `p_det` double-counts.
3. **BAND C's INTERNAL-DEFECT was unreachable** — a constant bias fails the accuracy-form GATE S for
   exactly the values BAND C tests. GATE S is now a slope/intercept regression.
4. **Two of four arms had targets a vacuous posterior hits exactly.** `H_GRID_41` flat =
   **0.7300000000** under *both* weight conventions — **the B-F1 mechanism survived the row #146
   correction untouched** — and `H_GRID_FULL` trapezoid flat = **0.6800000000** = δ−'s `h_gen`.
5. **The power transfer was biased in the orchestrator's favour** — `σ_seed` taken from
   floor-saturated B-SEL (0.0058) when unrailed arms give 0.0084–0.0230. **D-1's failure reproduced
   one cycle after A15 was adopted to end it.**

## 3. Standing constraints that bit this cycle — read before writing any registration

- **A15 is binding now.** It caught v1 twice. State every band's null and false-fail rate at the
  actual N; a control that cannot fail carries no verdict.
- **Verifier output is evidence, not authority.** This cycle's pre-checks were each decisively right
  *and* decisively wrong in the same report (one asserted b0's clean mean was ≈0.61 when it is
  0.7626; another demanded a "fix" to `_hpd_contains`, which matches the analytic Gaussian HPD in
  6/6 boundary cases). **Re-derive every decisive number before it reaches a ledger row.**
- **Subagent briefs must FORBID executing the registered measurement.** A synthesis agent
  dispatched to refute claims ran the A-7 re-score, which is why A-7 is recorded as an audited
  confirmatory recomputation rather than a blind measurement.
- Prereg-first for every measurement including free reads; scorers committed before their data;
  exoneration check (hard rule 1) **before** opening any thread — it saved a whole campaign this
  cycle (row #147).

## 4. Open author decisions

Nothing pending from the sentinel cycle — all 8 gate decisions are ruled and executed. Carried:

1. Systematics-budget **row 16** re-grade.
2. The **fix fork** for the base tilt — opens when C-SG returns.
3. **Landscape/T1 un-gate** (13 fused cells, ≈65 CPU-h). Gated by the author's row #128 ruling
   behind the base-tilt resolution, so the chain is: **C-SG → B-SEL verdict → fix fork → landscape**.
   T0 is banked; the 5 "off" cells are banked; the 13 fused cells were cancelled at 12.7 h of a 14 h
   wall.
4. **New (row #146 item 6):** row #144's residual bound (≥0.073) was derived against −0.112 and
   needs recomputing against **−0.1083**. Direction is small and unfavourable; **not** asserted to
   survive.

## 5. Operational

`--cpus-per-task=2` for correspondence arms; 5 h walltime and expect a straggler tail; subagents
must block in the foreground. **C-SG cost is ≈51–69 CPU-h, not the ≈35 v1 claimed.** Workspace
expires **2026-09-23** with 104 GB on it — `ws_list` says 34 days and the cluster skill documents
`ws_extend emri 60`, contradicting runbook 23's "0 extensions"; the extend/archive call is the
author's. The 130 per-event CSVs are already safe locally with a SHA-256 manifest.

## 6. Resume recipe (one line)

Read rows #145–#148 → run §9's free `L_cat_no_bh ≡ 0` decomposition → if it clears, run the 4-seed
pilot and set the bands from `σ̂_seed` → then implement C-SG per §2 (design B) and run the 46 seeds.
