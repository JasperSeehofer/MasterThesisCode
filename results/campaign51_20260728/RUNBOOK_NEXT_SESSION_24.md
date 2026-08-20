# Runbook — next session (written 2026-08-20, supersedes RUNBOOK_NEXT_SESSION_23)

**Read first:** ledger row **#145**, then `PREREGISTRATION_1D_CORRESPONDENCE.md` AMENDMENT **A-7**
+ its VERDICT, then `docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md` (8 open
author decisions). Rows #127–#144 and `docs/RETROSPECTIVE_D1_20260820.md` remain the background;
do NOT redo any of it.

## 0. What changed on 2026-08-20 (after runbook 23)

Runbook 23 §1 registered one next step: build a positive control that can fail. Before building
it, a zero-compute forensic on the banked arm JSONs found **why the existing one could not**.

**The defect.** `correspondence_1d.py:1965`/`:2479` floor a zero per-event likelihood **in log
space** at `-1.0e300`. Correctly stated (narrowed by measurement): this is *numerically identical
to correct `-inf` whenever ≥1 grid node survives* — `max|Δ mean_h| = 0.000e+00` across all 98 such
banked seeds. It bites only when **every** node is masked, where correct `-inf` fires the harness's
own `isfinite().any()` guard but the sentinel banks a finite, normalizable posterior **silently**.

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

## 2. THE ROOT CAUSE IS UPSTREAM AND IS STILL OPEN

Every all-zero event has `L_cat_no_bh = 0`, `B_num = 0` **and `g_frac = NaN`** — an empty candidate
set / undefined catalogue–completion mixing fraction — in **100%** of cases, against a 3–6%
baseline. 25/70 catalogue-mode seeds, **0/60 population-mode**. **This is a generator/data defect,
not a numerical one.** The proposed fixes stop the harness silently banking a fabricated posterior
when it happens; they do not stop it happening.

**Why the mirror places a host in the catalogue that the ball-tree lookup then fails to recover is
the open question**, and gate-presentation item 7 proposes it as the next thread **ahead of** the
row #144 §6 positive control — because that control would inherit the defect.

## 3. Open author decisions (8, all in the gate presentation §6)

`docs/derivations/GATE_PRESENTATION_SENTINEL_COMBINE_20260820.md`:
**[DO]** 1 sentinel fix · 2 trapezoid-weight fix · 7 open the `g_frac`-NaN thread first.
**[RULE]** 3 corrected numbers supersede published · 4 all catalogue-arm rails are artefacts ·
5 B-F1's "truth to four decimals" withdrawn, the control fails · 6 G-1's PASS recorded UNSUPPORTED.
**[RULE]** 8 **amendment A15** — still pending from row #144 §7.

Carried over from runbook 23 §5, still open: systematics-budget **row 16** re-grade; the **fix
fork** for the base tilt; whether to **un-gate the landscape/T1 round**.

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

Read row #145 → A-7 verdict → the gate presentation's 8 decisions → get the author's ruling → then
either implement the two fixes or open the `g_frac = NaN` empty-candidate-set thread, and only
after that build the row #144 §6 positive control.
