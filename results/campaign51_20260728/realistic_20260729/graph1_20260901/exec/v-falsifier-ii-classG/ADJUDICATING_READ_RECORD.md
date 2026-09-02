# v-falsifier-ii-classG — ADJUDICATING READ RECORD (falsifier (ii), post-A′ 33-seed fleet)

**Date:** 2026-09-02 · **Agent:** wave-2 decisive verifier (k-falsifier-ii-fleet adjudicating-read
slot, Research Graph 1 Branch E) · **Independence:** this verifier built nothing this fleet uses —
not the A′ implementation (`2b657255`), not the fleet driver, not the submission (job 6769177).

**Class:** post-data adjudicating read of a registered falsifier. **This document is EVIDENCE for
the author's `d-a4-final-ratification` [RULE] — it ratifies nothing and promotes no claim.**
Every band call below returns to the author.

**Registered test (spec of record):** `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.1(ii) — on the
class-G venue with rung 1 repaired in the Option A′ form, the registered v2.9 conditional
prediction must land: **LHS2(bt) = 0.00740040 ± 0.00024951, band ±3σ_comb two-sided, AND the G4
arm-coherence ratio ∈ [0.8613, 0.8675]** (v2.9 text:
`PREREGISTRATION_P3_2D_REPAIR_20260827.md:1023-1032`; G4 interval registration `:969-975`;
band form "3σ_comb plainly, σ_comb = sqrt(σ_pred² + σ_new²), σ_pred frozen, σ_new realized"
`:787-790` / PA-2DR-2).

---

## 1. Fleet completion and provenance (verified, not assumed)

- **sacct, job 6769177 (array 0–32):** `sacct -j 6769177 -X` → **33 × `COMPLETED 0:0`**, no other
  state (verified this session, 2026-09-02; tasks ended 13:21–13:24 CEST, ~33 min each).
- **Out-root** `$WS/p3_2d_fleet_aprime_20260902`: **66/66 `<arm>_<seed>_meta.json` present**
  (bt+bc × seeds 900101–900133), matching the LAUNCH RECORD's registered configuration.
- **Commit stamp uniformity (all 66 metas):** `git_commit = c83e391d…` (66/66),
  `tree_dirty_incl_instrument = "clean"` (66/66). Verified locally:
  `git merge-base --is-ancestor 2b657255 c83e391d` → **true** — the fleet ran at a commit
  containing the landed Option A′ repair, i.e. the falsifier's registered venue ("rung 1 repaired
  in the Option A′ form") actually existed in the code that ran. This is the provenance
  distinction from the pre-repair banked fleets (6723958/6730213 at `d04d9dc9`).
- **Arm engagement:** a22 stamps show `catalogue_numerator_survival_2d = "mz_sel"` on all bt
  metas and `"off"` on all bc metas (uniform per arm); the two arms' realized LHS2 differ by
  ~16 % — the twin axis engaged, neither arm silently ran the other's flag.
- **Catalogue pin:** `catalogue_pin_ok: true` in the metas (driver aborts on md5 mismatch;
  pin `c52c13b5…`, confirmed at launch per the LAUNCH RECORD).
- **Retrieval for the independent re-derivation:** 66 metas + 66 per-seed diagnostics CSVs + the
  2 registered stage stdouts were tarred on the cluster with an md5 manifest and verified locally:
  **134/134 md5 OK** (no symlinked shared-pool content in the bundle — explicit file list, per the
  row #311 gotcha).

## 2. Registered post-processing (the registered computation, not a reimplementation)

Run by this verifier on the cluster login node at the fleet's own checkout
(`~/darksiren-emri` @ `c83e391d`, tracked tree clean — verified before running), exactly the
sibling-readout pattern (`P3_2D_REPAIR_READOUT_20260828.md` §1):

```
python results/…/p3_2d_fleet.py --stage lhs2d --arm bt --seeds 900101,…,900133 --out-root $WS/p3_2d_fleet_aprime_20260902
python results/…/p3_2d_fleet.py --stage lhs2d --arm bc --seeds 900101,…,900133 --out-root $WS/p3_2d_fleet_aprime_20260902
```

(stdout archived in the out-root as `lhs2d_{bt,bc}_33seed_stdout.json`). One operational note,
disclosed: the login-node venv needs `source cluster/modules.sh` first (cluster skill gotcha —
libpython3.13); the first invocation without it failed before any computation and was re-run.
`C2_star = 0.06124403326364123` read from the registered companion
(`ca_rhs_work2d/p3_2d_companion_v2.json`, the PA-2D-3 v2 segment-aware pass — never v1).

**Registered stage output (33 seeds, verbatim):**

| quantity | bt (twin) | bc (coded) |
|---|---|---|
| `LHS2_mean` (D1+D2, the registered read) | **0.007446893040944427** | **0.00642600889201313** |
| `LHS2_sem` | **0.0002055871033945423** | 0.00019619784335435467 |
| `LHS2_D1only_mean` | 0.006927246698101411 | 0.005906362549170113 |
| `LHS2_paired_ratio_mean ± sem` | 1.076230 ± 0.009641 | 1.089416 ± 0.011404 |
| `dead_row_identity_all_ok` (P5/G6 form) | **true** (33/33) | **true** (33/33) |
| `pa2dr7_fraction` (G5 form) | **0.0** | **0.0** |

## 3. Independent re-derivation (verifier's own arithmetic — "verifier output is evidence")

From the raw per-seed diagnostics CSVs (md5-verified local copies), with independently written
code (pandas/numpy, no import of the fleet driver): rows at h ≈ 0.73,
`w2 = a2/(a2+B_num_wbh)` with the zero-default masked divide (`a2 = alpha_G_phi·L_cat_with_bh`),
`LHS2_s = (C2*/200)·Σ_acc(1−w2)`, fleet mean ± SEM (ddof=1), per arm over the 33 seeds.

**Result: exact match to the registered stage output — relative deviation 0.0 on all four
decisive numbers** (bt mean, bt SEM, bc mean, bc SEM; script + full per-seed values:
scratch `aprime/rederive.py` / `rederive_result.json`, this session). The D1only means also
reproduce. G4 and the band arithmetic below are therefore double-derived.

## 4. Band evaluations (the registered §6.1(ii) reads)

### 4.1 LHS2(bt) vs the v2.9 conditional prediction

- Prediction: **0.00740040**, σ_pred = **0.00024951** (frozen; v2.9,
  `PREREGISTRATION_P3_2D_REPAIR_20260827.md:1029-1031`; ladder
  0.00500770 × 1.1585 × 1.1944 × 1.0680 = 0.00740040, `:716/:727`).
- σ_new = realized fleet SEM = **0.00020559** (33 seeds — **below** the frozen planning scale
  σ_pred, so the §6.1(ii) null-power clause is satisfied; no UNDERPOWERED disposition arises).
- **σ_comb = sqrt(0.00024951² + 0.00020559²) = 0.00032330**; band = ±3σ_comb =
  **±0.00096989** (PA-2DR-2 band form, ε provably inert).
- Realized: **LHS2(bt) = 0.00744689 ± 0.00020559**.
- Deviation: **+0.00004649 = +0.144 σ_comb** → **INSIDE**, nowhere near either band edge
  (edge is at 3.0 σ_comb; standing rail/edge convention not triggered).

### 4.2 G4 arm-coherence

- Registered form: ratio of fleet means, coded/twin = LHS2(bc)/LHS2(bt) (the P4/P1 form of the
  sibling readout §2/§7); registered interval **[0.8613, 0.8675]** (v2.6 G4, spanning both
  derivation routes; `PREREGISTRATION_P3_2D_REPAIR_20260827.md:969-975`).
- Realized: **G4 = 0.00642601/0.00744689 = 0.862911 → INSIDE** the interval.
- Edge-proximity disclosure (REPORTED-ONLY, not a call): the point ratio sits 0.00161 above the
  lower edge (26 % of the 0.0062-wide interval from it; the historical values were 0.865491 at
  24 seeds and 0.866484 at 33 seeds pre-repair). A paired per-seed ratio companion (this
  verifier's own arithmetic, not a registered read) gives 0.86090 ± 0.00472 — the registered
  point read is inside, and the gate as registered defines no tolerance beyond the interval
  itself, so the stamp is INSIDE; the proximity is disclosed for the author's eyes.

## 5. Gates

| gate | result |
|---|---|
| G5 form (`pa2dr7_fraction`) | **PASS** — exactly 0.0, both arms, all 33 seeds (the ×1.1944 factor transfers) |
| G6/P5 form (`dead_row_identity_all_ok`) | **PASS** — exact to float round-off on **66/66** arm-seed pairs |
| Engagement / A21-style scope | **PASS** — arm flags uniform and distinct per arm (§1); fleet ran at the A′ commit; fresh out-root (no PA-CA-11 reuse: all 66 metas freshly written by job 6769177, timestamps 13:21–13:24) |
| GATE M2-LINK parts (ii)/(iii) | **PASS / monster-class ABSENT** (local re-derivation, 66/66) — §6 |
| GATE-ACC (F12, reporting-only) | cluster stage run in flight at finalization — §6/§8 |

## 6. `--stage gates` (GATE-ACC re-check + M2-LINK + F10c) — first fleet-level run in this line

The A′ implementation record (§6.3 item 3) flags the GATE-ACC re-check as the open R4/fleet item.
Checked directly: **no `gates_*.json` exists in either banked out-root**
(`p3_2d_fleet_20260825`, `p3_2d_fleet_repair_20260827`) — the banked readouts used the dev gates
G1–G3 plus the lhs2d-derived G4–G6; `stage_gates` had not previously been run on any fleet. This
verifier ran it on the post-A′ fleet (both arms, 33 seeds).

**Standing instrument characteristic found while preparing this (pre-adjudicated as
common-mode, NOT an A′ regression):** every meta's `rhs_f2_provenance_bitcheck` (M2-LINK part i,
the F11(i) "bit-level" CSV round-trip check, pass iff `max_rel_dev == 0.0`) reads
`pass: false` at `max_rel_dev` 1.5e-15…2.8e-13 on this fleet — **and 66/66 `pass: false`
(max_rel_dev ~1e-14) on the banked pre-repair repair-fleet metas as well**, i.e. the strict
`== 0.0` predicate was already unattainable in the fleet that banked CONFIRMED (row #216).
The deviation is float-text round-trip round-off, orders of magnitude below every band in play.
Consequence: `M2_LINK_all_pass` is `false` on every fleet this instrument has ever written, for
this part-(i) reason alone; parts (ii) (Mahalanobis) and (iii) (monster absence) are the
substantive halves and are reported below.

**Results.** The full `--stage gates` invocation (launched by this verifier, both arms, 33
seeds, logs + `gates_*.json` land in the out-root) was still inside its initial
`build_b0i_2d_selection_objects` phase at 1h16m CPU (single-threaded login-node build; process
healthy, 98 % CPU, no per-seed output yet) at finalization time. Because F12 registers GATE-ACC
as **REPORTING-ONLY** ("0.5821 is NOT a reference"; no PASS/FAIL threshold) and §6.1(ii)
conditions on the LHS2/G4 bands only, the adjudication below does not hinge on it; its numbers
are an addendum item (§8). The M2-LINK substantive parts were computed by this verifier
LOCALLY over all 66 arm-seeds from the md5-verified CRB + diagnostics CSVs (66/66 CRB md5 OK),
with the driver's own registered formulas re-implemented independently:

- **Part (i) provenance columns/round-trip:** all 12 `PROVENANCE_COLUMNS` present, 66/66. The
  meaningful bit-level half is the meta-level `rhs_f2_provenance_bitcheck` adjudicated above
  (`false` at 1.5e-15…2.8e-13, common-mode with the banked fleet). Code-reading note for the
  author: `stage_gates` passes the re-read CSV as `events`, so its own part-(i) compares the CSV
  with itself and passes trivially — the meta-level check at fleet time is the real one.
- **Part (ii) Mahalanobis² fleet bound: PASS, 66/66 arm-seeds.** max Mahalanobis² = **21.745**
  vs threshold χ²₂(1−1e-3/200) = **24.412**; zero singular covariances. Identical across arms
  (expected: bc/bt share identical drawn latents per seed — the arm flag changes only the
  estimator numerator, not the draw).
- **Part (iii) monster-absence (−50 nats): the registered monster class is structurally
  ABSENT — zero FINITE sub-−50-nat live rows, 66/66 arm-seeds.** The coded predicate as
  written additionally counts 19 rows/arm at ln-ratio = −∞: these are `L_cat_with_bh = 0`
  rows, i.e. exactly the pre-registered F16/D2 **dead-row** class (a subset of the 56
  dead rows/arm the lhs2d stage accounts for with summand 1 and the exact G6 identity —
  which passes 66/66). The banked pre-repair 33-seed fleet (out-root
  `p3_2d_fleet_repair_20260827`, the fleet that banked CONFIRMED, row #216) shows the
  IDENTICAL characteristic: 0 finite / 20 −∞ per arm (computed the same way this session).
  Adjudicated: **no monster signal; the −∞ count is the known dead-row population,
  common-mode with the banked comparand** — flagged (with part (i)'s `== 0.0` strictness)
  as a coded-predicate vs registered-intent hygiene item, not a data defect. Full
  decomposition: scratch `aprime/monster_decomposition.json` / `m2link_local.json`.
- **GATE-ACC (F12):** pending the cluster stage run (reporting-only); per-seed p̄_s /
  `n_drawn_total` will be in `gates_{bt,bc}.json` when it completes — the A′ presentation's
  AR-3 L9 expectation (~3 % acceptance move, orders of magnitude of headroom) is NOT
  independently re-verified here and is not claimed.

## 7. Mechanical outcome (per §6.1(ii)'s own semantics — no promotion)

| registered read | realized | band | call |
|---|---|---|---|
| LHS2(bt) | 0.00744689 ± 0.00020559 | 0.00740040 ± 0.00096989 (3σ_comb, σ_comb = 0.00032330) | **INSIDE (+0.144 σ_comb)** |
| G4 | 0.862911 | [0.8613, 0.8675] | **INSIDE** |

**Mechanical outcome: INSIDE-BOTH.** Per §6.1(ii)'s registered semantics
(`PROPOSAL_2D_TWIN_ADOPTION_20260829.md:270-275`): *"inside both ⇒ the ladder model is complete
to rung 3 and the remaining ×1.96 is venue-side until shown otherwise (attribution stays
provisional)"* — i.e. for **c-a4-structural** this read **does not refute** the attribution and
supports the ladder's completeness through rung 3; the outside-LHS2 branch (calibration status
drop to `derivation-only`) and the outside-G4 branch (REFUTED-AS-CALIBRATED return) are **not**
triggered. The §v2.7 cap stands: nothing here upgrades any status beyond `supported`.

**Companion (REPORTED-ONLY):** realized X = RHS₂/LHS₂ cannot be recomputed here without a fresh
RHS₂ pass (not registered for this falsifier and not run); the ladder-implied residual at the
realized LHS2(bt) is 0.01451300/0.00744689 ≈ 1.949 if the banked 33-seed RHS₂ is carried over —
carried-over, not fresh; stated only to orient the author, with no band and no verdict on it.

**What this read does NOT do:** it does not ratify A4 (that is `d-a4-final-ratification`, the
author's [RULE]); it does not close the row #211 PARK; it does not test rung-1's factor in
isolation (the fleet realizes the full repaired law, and the prediction tested is the composed
three-rung ladder point); it does not certify the twin `verified` (v2.7 cap).

## 8. Absences and blocked items (named, per the verdict discipline)

1. **No fresh RHS₂ pass** — X = RHS₂/LHS₂ at the repaired venue is not measured by this read
   (not part of the registered falsifier; would need the ca_rhs scorer re-run, unauthorized).
2. **M2-LINK part (i)** strict predicate is unattainable at float-text round-trip precision on
   every fleet to date (common-mode; flagged as registration hygiene, §6).
3. The first `--stage gates`/`--stage lhs2d` invocation of this session failed on the login-node
   libpython environment before computing anything; re-run with `cluster/modules.sh` (§2) — no
   partial output survived into the archived stdouts.
4. **GATE-ACC numbers are the one ABSENT item**: the cluster `--stage gates` run (both arms
   chained) was still in its selection-object build phase (>1h16m CPU, healthy) at
   finalization; `gates_bt.json`/`gates_bc.json` will land in the out-root and belong in an
   addendum here. F12 registers it reporting-only; no §6.1(ii) band conditions on it, so the
   §7 mechanical outcome is unaffected by its absence. M2-LINK's substantive parts (ii)/(iii)
   were adjudicated from the raw data locally (§6) and do not wait on it.

---

*Every decisive number above appears twice: once as the registered instrument's verbatim output
(archived stdouts / gates JSONs in the out-root) and once re-derived by this verifier's own
arithmetic from md5-verified raw per-seed CSVs, with exact agreement. Scratch analysis under this
session's scratchpad (`aprime/`); no code was edited, nothing committed by this verifier.*

---

## Addendum stamp (session close, 2026-09-02 ~15:30 CEST)

At this verifier's session close the cluster `--stage gates` chain (bt then bc) was still
running and healthy (PID 250133, 98 % CPU, 2h16m CPU time, single-threaded, still inside the
initial `build_b0i_2d_selection_objects` phase — no per-seed GATE-ACC output yet). When it
completes it writes `gates_bt.json` then `gates_bc.json` (and `gates_33seed.DONE`) into
`$WS/p3_2d_fleet_aprime_20260902/` with logs `gates_{bt,bc}_33seed.log`. Whoever picks this up:
those JSONs carry the per-seed GATE-ACC p̄_s / n_drawn_total (F12 REPORTING-ONLY) and the
driver's own M2-LINK dicts (whose part (i) self-compares the CSV, §6). Nothing in §7's
mechanical outcome waits on them.
