# Campaign report — production-2D closure, day 2 (2026-08-19)

**A7 comprehension-first readout.** Front of record: row #127 (closure + landscape).
Everything below is production-native unless explicitly marked harness/class-level (P7-4's
venue-scoping spine is respected throughout). Every branch labeled [RULE] awaits the
author's ratification; nothing has been adjudicated on the author's behalf.

## 1. What we asked, and what we now know

The question of record: **what owns the production 2D offsets** Δ = +0.054 (iiib) /
+0.067 (joint_r1)? After today, the answer landscape is much sharper — mostly by
elimination, which is the story of the day:

1. **Event-draw luck does not own it** (T0, banked yesterday): z = 4.75/5.53; jackknife-889
   ROBUST.
2. **The documented Eddington-in-M treatment moves it the WRONG way**: its −0.020 means the
   unexplained systematic residual is r = Δ − (−0.020) = **+0.074 / +0.087**, vs
   2·σ_total ≈ 0.023 (σ_boot 0.0114/0.0121; s_Edd carried as a point value, as registered).
   **Budget branch: B-UNOWNED** in both venues [RULE] — a residual ~3–4× its uncertainty
   remains after all registered production-native legs.
3. **The selection-fusion lever does not own it** (row #119, prior).
4. **NEW — the catalogue-leg mass overlap does not own it.** Today's registered production
   counterfactual (jobs 6369297–6369304, 250/250 clean, all gates PASS) neutralized every
   candidate's mass overlap (V1′, measure-coherent) and the 2D posterior mean barely moved:
   **ΔV1 = +0.0010 / +0.0032** — positive, and far below the 0.006 materiality line. The
   registered branch is **C-MIXED** by the letter (joint_r1 sits at 0.0032, inside the
   registered 0.003–0.006 gap; iiib is squarely C-REFUTED) [RULE]; the substantive reading
   is that catalogue-leg mass-overlap ownership is refuted at materiality in both venues.
   The leg is genuinely alive — inflating σ_M ×2 moves the mean by +0.009/+0.016 — its
   production operating-point contribution is simply ≈ 0.
5. **The per-event structure was real but was structure-around-the-shift, not the shift.**
   The morning's registered regression (R-MIXED [RULE]) found the 2D−1D slope excess
   concentrates in catalogue-supported, impostor-borne events — and the counterfactual now
   shows that removing exactly that machinery leaves the mean in place. The regression's own
   P8 caveat anticipated this: covariates explain the variation around a near-universal
   positive shift carried by ~95% of events.

**Net: the +0.074/+0.087 residual lives in the 2D channel's remaining structural
difference from 1D — the COMPLETION-leg mass factor (g_i/g_frac geometry) and/or the
alpha_G_phi-path asymmetry — not in the per-candidate catalogue mass overlap.**

## 2. Mechanism register after today

- **M-B (catalogue→completion re-balance):** refuted in DIRECTION by the regression (both
  venues, sensitivity-stable).
- **M-A (catalogue-kernel inverse-mass shift):** the derivation's sign/coupling logic stands
  (F1/F2 are real properties of the kernel), but its production MAGNITUDE at the catalogue
  leg is now refuted by V1′ — consistent with the derivation's own honest gap (§3: the
  within-kernel form caps at ~σ_z², short of the offset).
- **M-C (φ-support truncation) and the completion-leg F1/F2 analog:** now the leading
  candidates — both live in the completion leg's g_i(z;h) geometry, which V1′ deliberately
  left untouched and which carries the V-prod off-class venue bias (+0.008…+0.015
  descriptive) plus the h-dependent μ_cond/φ-edge structure the mechanism doc describes.
- The mechanism doc's blind T2 landscape predictions remain registered and untouched
  (the fused cells never ran — see §3).

## 3. The landscape round: cancelled and gated (author [RULE], on the record)

Job 6364821 was cancelled at 12h45 (author-directed): all 13 remaining cells were fused
cells, none finishable under the observed 18-worker contention (off cells ran 2.4–5.4 h vs
the ~5 h/cell fused estimate; the sizing anchor did not transfer). The author's gating
ruling — landscape/T1 re-runs only after the 2D residual is finally resolved — is recorded
verbatim in the closure prereg's CLUSTER FILL-IN 2 (commit 4128eab2). The five banked off
cells remain fully registered reads (1D off-basis ladder + V-prod off). The closure's
execution-completeness clause is amended accordingly.

## 4. Decisions table for the author

| # | Tag | Decision |
|---|---|---|
| 1 | [RULE] | Ratify the budget branch: **B-UNOWNED**, r = +0.074/+0.087, σ_total 0.0114/0.0121 per venue (P7-8: one realization). |
| 2 | [RULE] | Ratify the regression branch **R-MIXED** as recorded (M-B direction-refuted; catalogue-leg per-event structure; σ_M leg UNDERPOWERED-NULL). |
| 3 | [RULE] | Ratify the counterfactual branch **C-MIXED** with the ownership-refuted-at-materiality reading (or amend the reading). |
| 4 | [DO] | Open the **completion-leg counterfactual** (next elimination step, same pattern: neutralize/deform g_i in a registered instrument; prereg-first, verifier pre-check, physics gate returns to you). ~12 CPU-h class. |
| 5 | [RULE] | Fix-fork status: with the catalogue-leg refuted, forks (a)/(b) as originally scoped (catalogue-kernel repair) are DEAD; the fork re-opens against whatever the completion-leg test finds; (c) document-as-systematic remains the fallback. Confirm this re-scoping. |
| 6 | [DO] | Housekeeping: prune `backup-pre-excise-20260819` + `refs/original` and `git gc` (~4 GB) — the rewrite is pushed and mapped (`docs/HISTORY_REWRITE_20260819.md`). |

## 5. Ops appendix (for the record)

- History rewrite executed and pushed (author-approved option 2): 16 unpushed commits
  excised of the two 4 GB staging CSVs; freeze d6fc1ccf → 26bcd9a4; cluster repo re-synced,
  tag re-pointed; origin fully in sync from `116ccd3a` onward; pushes now take seconds.
- Counterfactual instrument: `[PHYSICS]` commit (gate-ledger rows appended), flag default
  bit-identical to production (N-0 measured ≤ 6.2e-14), 22-test suite, full suite
  1568 passed.
- Fleet economics: 250 tasks, ~3 min each, all COMPLETED first pass; total ≈ 12.5 CPU-h —
  the same afternoon that the 13-cell harness round (~160 CPU-h class, non-finishing) was
  cancelled. The production-direct pivot cost ~1/10th and answered a sharper question.
- Workspace expiry 2026-09-23 (0 extensions) — unchanged; finals are on local disk + git.
