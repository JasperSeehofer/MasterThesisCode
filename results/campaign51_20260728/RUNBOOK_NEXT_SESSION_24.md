# Runbook — next session (written 2026-08-20, supersedes RUNBOOK_NEXT_SESSION_23)

**Read first:** ledger rows **#145 → #146 → #147** (and the two addenda to #145), then
`PREREGISTRATION_1D_CORRESPONDENCE.md` AMENDMENTS **A-7** and **A-8** with their VERDICTs. Rows
#127–#144 and `docs/RETROSPECTIVE_D1_20260820.md` remain the background; do NOT redo any of it.

**All 8 decisions of `docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md` §6 are RULED
and EXECUTED** ("please continue, approved"). **The next front is the row #144 §6 positive
control — it is UNBLOCKED (row #147).**

## 0. What changed on 2026-08-20 (after runbook 23)

Runbook 23 §1 registered one next step: build a positive control that can fail. Before building
it, a zero-compute forensic on the banked arm JSONs found **why the existing one could not**.

**The defect.** `correspondence_1d.py:1965`/`:2479` floor a zero per-event likelihood **in log
space** at `-1.0e300`. Correctly stated (narrowed by measurement): this is *numerically identical
to correct `-inf` whenever ≥1 grid node survives* — `max|Δ mean_h| = 0.000e+00` across all 98 such
banked seeds. It bites only when **every** node is masked, where correct `-inf` yields **NaN**
statistics (visibly broken) but the sentinel banks a finite, normalizable posterior **silently**.
(An earlier wording claimed an `isfinite().any()` guard already existed — it did not; corrected in
the second addendum to row #145. One has now been added as part of the approved fix.)

**STATUS 2026-08-20, after the author's "please continue, approved":** both fixes are
**IMPLEMENTED** (row #146) and A15 is **ADOPTED**. Legacy paths are retained and reproduce the
banked fleet **123/123** bit-exactly. The **fully-corrected** numbers (both fixes) are in row #146
item 5 and supersede A-7's combine-only table: b0 **+0.0296** · bsig005 **+0.0362** · eden05
**+0.0139** · eden2 **+0.0321** · bf1 **+0.0358** · bout **−0.1287** · bsel **−0.1083** · bself
**−0.1126** · bden **−0.1159**. The bisection *signal* is preserved (successive differences
−0.0043, −0.0033 vs published −0.0043, −0.0030). **Open:** row #144's residual bound (≥0.073) was
derived against −0.112 and needs recomputing against −0.1083. **Next: item 7, the `g_frac = NaN`
thread (§2).**

**Blast radius 25/123 banked seeds (20.3%)** — catalogue-mode 25/70, population-mode **0/53**.
21 are exactly flat ⇒ `mean_h` = the `H_GRID_41` midpoint `(0.600+0.860)/2 = 0.7299999999999999`,
**which coincides with `H_TRUE`**; `map_h = 0.600` (argmax tie-break) ⇒ `r_low = True`; and
`c50=c68=c90=True`. One seed reports *unbiased*, *railed* and *covered* at once, from grid geometry.
The other 4 are non-uniformly masked and spuriously informative (0.8087–0.8400).

**Zero-compute recovery.** 130/142 arm work-roots retained `event_likelihoods.csv`; retrieved
(151 MB) to `results/prod2d_closure_20260818/arm_event_likelihoods/` with SHA-256 manifest +
provenance stamp. The whole fleet was re-scored **without re-running ~150 CPU-h**. Scorer:
`results/prod2d_closure_20260818/rescore_sentinel.py`; output JSON alongside it.

## 1. Corrected numbers of record (A-7 verdict)

| arm | N | published | corrected (physics-floor) | Δ | R_low | C68 |
|---|---|---|---|---|---|---|
| b0 | 25 | +0.0245 | **+0.0298 ± 0.0046** | +0.0053 | **0.36 → 0.00** | 0.64 → 0.48 |
| bsig005 | 23 | +0.0348 | +0.0366 ± 0.0072 | +0.0019 | 0.17 → 0.00 | 0.43 → 0.39 |
| eden05 | 10 | +0.0093 | +0.0140 ± 0.0044 | +0.0046 | 0.10 → 0.00 | 0.90 → 0.80 |
| eden2 | 10 | +0.0211 | +0.0322 ± 0.0088 | +0.0111 | **0.50 → 0.00** | 0.60 → 0.30 |
| **bf1** | 2 | −0.0000 | **+0.0359 ± 0.0036** | **+0.0359** | **1.00 → 0.00** | **1.00 → 0.00** |
| bout / bsel / bself / bden | 15/12/11/15 | −0.1293 / −0.1120 / −0.1163 / −0.1193 | identical | **0** | unchanged | unchanged |

**The means barely move; the RAIL and COVERAGE statistics are artefacts.** `R_low` → exactly 0.00
in every catalogue-mode arm. **B-F1, the positive control, does not merely carry no information —
corrected, it FAILS** (+0.0359, coverage 0/0/0; PROVISIONAL at n = 2, and a poor control anyway:
under `f ≡ 1` **100% of its events have `g_frac = NaN`**).

**Two of the orchestrator's own claims were REFUTED by measurement and withdrawn:** B-OUT's rail is
*not* sentinel-manufactured (physics-floor moves it ≤1.1e-16; **row #139 stands**), and the zeros
are *not* underflow (fleet min non-zero 4.876e-48, ~302 orders above the subnormal floor).

**Unaffected:** the bisection chain B-SEL/B-SELF/B-DEN (Δ ≡ 0 to ≤1.0e-15) ⇒ **row #140's
PROVISIONAL-WITH-A-BOUND is untouched**; and all production numbers (the additive sentinel exists
nowhere else, and no production module imports these functions).

## 2. THE ROOT CAUSE — mirror-specific, h-INERT, and CLOSED (row #147)

Every all-zero event has `L_cat_no_bh = 0`, `B_num = 0` **and `g_frac = NaN`** — an empty candidate
set / undefined catalogue–completion mixing fraction — in **100%** of cases, against a 3–6%
baseline. 25/70 catalogue-mode seeds, **0/60 population-mode**. **This is a generator/data defect,
not a numerical one.** The proposed fixes stop the harness silently banking a fabricated posterior
when it happens; they do not stop it happening.

**MEASURED, same session: the defect does NOT reach production.** All five banked production
diagnostics (postfix iiib/joint_r1, frozeng iiib, battery v0_iiib, counterfactual v0_iiib; 1588
events each) have **0 zero cells, 0 all-zero events and 0.0% `g_frac` NaN**. So neither the
sentinel nor its upstream trigger touched the post-fix baselines, the dark class 0.6001, or the
score −0.635 ± 0.017. See the addendum to ledger row #145.

**RESOLVED, row #147 (AMENDMENT A-8), same session.** The thread was opened with its exoneration
check first (hard rule 1: §2 item 8 records the #29 zero-host fallback as h-inert per #55) and
closed in one zero-compute pass:

- Excluding every all-zero event changes its seed's `mean_h` by **0.000e+00** — exactly, across all
  25 seeds / 41 events. The exoneration is confirmed at machine precision; these events **cannot**
  carry bias.
- **BAND G → CONTROL-SAFE.** The mechanism is confined to the catalogue-mode draw (**0/60**
  population-mode seeds), and **B-SEL — which draws from the estimator's own detected-dark density
  and is the closest analogue to the planned control — has 0 all-zero events in 12 seeds.**
  **The row #144 §6 positive control is UNBLOCKED and is now the next front.**
- Two populations: **B-F1 is structural** (`f ≡ 1` ⇒ `g_frac` undefined for 100% of its events);
  the real-completeness arms have a 3–5% `g_frac`-NaN baseline. **Fidelity gap, named:** the mirror
  emits hostless events at 3–5%, production at **0.0%** (0 in 1588).
- **Left OPEN deliberately:** *why* `B_num = 0` (complete pixel vs collapsed integration bounds) is
  undecidable from banked data — the declared structural blindness — and the CONTROL-SAFE branch
  does not require it. Documented as a mirror-fidelity limitation, not a mechanism.

## 3. Decisions — all 8 RULED and EXECUTED (nothing pending from this cycle)

`docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md` §6, author ruling "please
continue, approved":
**[DO]** 1 sentinel fix **DONE** · 2 trapezoid-weight fix **DONE** (row #146) · 7 `g_frac`-NaN
thread **OPENED AND CLOSED**, CONTROL-SAFE (row #147).
**[RULE]** 3 corrected numbers supersede published **RATIFIED** · 4 all catalogue-arm rails are
artefacts **RATIFIED** · 5 B-F1's "truth to four decimals" withdrawn, the control fails
**RATIFIED** · 6 G-1's PASS recorded UNSUPPORTED **RATIFIED** · 8 **A15 ADOPTED** into
`docs/RESEARCH_CYCLE.md`, with the NULL-BY-CONSTRUCTION corollary.

**Still open, carried from runbook 23 §5** (author decisions, NOT covered by the above):
systematics-budget **row 16** re-grade · the **fix fork** for the base tilt · whether to **un-gate
the landscape/T1 round**. **New, from row #146 item 6:** row #144's residual bound (≥0.073) was
derived against −0.112 and needs recomputing against the corrected **−0.1083**.

## 4. Method lessons banked this session

- **A15 is not academic.** The orchestrator's own A-7 draft twice reproduced the failure A15
  forbids: a band calibrated against the wrong null (`2·SE` for a *paired* difference whose
  sampling variance is exactly 0), and a BAND O whose two branches were **both pre-determined**
  (withdrawn as undecidable). A registration needs its own adversarial pre-check — A-7's returned
  **NOT-READY** with ten required amendments, all applied before the verdict was read.
- **Verifier output is evidence, not authority.** The pre-check's `b0 clean mean ≈ 0.61` was wrong
  (it is 0.7626), and its third proposed fix (`_hpd_contains`) was wrong — the routine matches an
  analytic Gaussian HPD in 6/6 boundary cases. Both were caught by re-deriving, not by trusting.
- **A can-fail control can be free.** GATE R-0a (re-run the *defective* path and demand
  bit-reproduction of all 123 banked seeds) passed 123/123 and simultaneously proved CSV↔JSON
  pairing. R-0b (no-op identity on the 79 sentinel-free seeds) passed 79/79.
- **Process violation, disclosed:** a synthesis agent dispatched to *refute* claims ran the primary
  measurement before A-7 was finalized. A-7 is an **audited confirmatory recomputation, not a blind
  measurement**. Subagent briefs must forbid executing the registered measurement.

## 5. Operational (unchanged unless noted)

`--cpus-per-task=2` for correspondence arms; generous walltime (5 h) and expect a straggler tail;
subagents must block in the foreground. **Workspace expires 2026-09-23** — `ws_list` reports 34
days remaining and the cluster skill documents `ws_extend emri 60`, which contradicts runbook 23's
"0 extensions"; 104 GB of campaign data is at risk and the extend/archive call is the author's.
The 130 per-event CSVs are already safe locally.

## 6. Resume recipe (one line)

All 8 decisions are ruled and executed (rows #146, #147). **Read rows #145–#147, then build the
row #144 §6 positive control** — an arm generated end-to-end by the estimator's own forward model
plus an injected-bias variant proving it detects a known displacement. It is UNBLOCKED, it is the
last thing standing between the campaign and a verdict on B-SEL's residual, and it must be
registered under A15 (its bands need stated operating characteristics, and the control itself must
be shown capable of failing).
