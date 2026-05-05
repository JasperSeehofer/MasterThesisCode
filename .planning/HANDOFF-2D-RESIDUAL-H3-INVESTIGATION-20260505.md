# Handoff — 2D channel residual after principled p_det fix (H3 investigation, 2026-05-05 16:00)

**Audience:** a fresh Claude Code session picking this up cold.
**Author:** the afternoon session that landed the principled-extrapolation fix.
**Status of caller:** all today's commits pushed; cluster validated post-fix; not running anything.

---

## TL;DR

Today's principled p_det extrapolation fix (`[PHYSICS]` commit `2b33cad`) **completely
resolved the 1D channel** at h=0.73 on the phase46-merged 1549-event CRB
(z dropped from −0.64 to **+0.19**, MAP=0.7309 within 0.001 of truth). It
**did not fully resolve the 2D channel**: 2D z dropped from +37 to +3.60
(bias halved from +0.0212 to +0.0141; σ_boot widened 6.5× to 0.0039).

**The user's framing** (which scopes the new session): adding the BH-mass
likelihood (the 2D channel) is *adding information* relative to the 1D
position-only likelihood. **Adding information should TIGHTEN the
posterior and bring MAP closer to truth, not farther.** The post-fix 2D
MAP being further from truth than the 1D MAP is therefore a structural
2D-specific bug, not a statistical residual.

**Investigation goal:** isolate and fix the structural mechanism in the
2D channel that makes adding M-information *increase* MAP bias relative
to position-only inference. The prime suspect is **H3: M_z vs M_source
coordinate mismatch in the 2D p_det grid query** (see §"Prime suspect"
below).

---

## What's already done (do not redo)

| Item | Commit / status |
|---|---|
| Tier 3 D(h) double-counting fix (2026-05-04) | `6754ddb` (intact; verified by test_22 today) |
| Phase 45 Wilson-LB anchor `(0, 0.7931)` removed | `2b33cad` |
| Phase 45 intermediate anchor `(0.05, 1.0)` removed | `2b33cad` |
| 1D `_zero_fill` body replaced with principled bridge | `2b33cad` |
| 2D `with_bh_mass_interpolated` body replaced with principled scheme | `2b33cad` |
| `_build_grid_1d` no longer prepends anchors | `2b33cad` |
| 15 new property-based tests (8 for 2D, 7 for 1D principled extrapolation) | `2b33cad` |
| `TestPhase45EmpiricalAnchor` (13 obsolete tests) deleted | `2b33cad` |
| 514/514 pytest pass; ruff + mypy clean | verified locally |
| Audit doc `.planning/2D-CHANNEL-AUDIT-20260505.md` | populated through Step 4 |
| `docs/H0_BIAS_RESOLUTION.md` §3.14 added | `2b33cad` |
| Memory: `feedback_principled_physics_choices.md` saved | (out-of-repo) |
| Cluster code synced (commit `2b33cad` on cluster) | `git pull` 2026-05-05 ~15:15 |
| Post-fix h=0.73 closure on phase46-merged 1549 events | job 4229895, COMPLETED |
| Post-fix posteriors at `simulations/cluster_run_closure_h0p73_postfix_finegrid/` | rsynced; analyzer run |
| Post-fix verdict JSON `outputs/phase46_merged/h0p73_postfix_verdict.json` | written |
| Wiki debrief filed (W-LOOP-03, W-PRE-06, 3 patterns, 1 incident, EXP-20) | vault commit `86dff7f` |

---

## Post-fix verdict numbers (h=0.73 phase46-merged, 1473 detected events)

```
1D:  MAP=0.7309  bias=+0.0009  σ_boot=0.0046  z=+0.19  pos_frac=0.72   PASS ✅
2D:  MAP=0.7441  bias=+0.0141  σ_boot=0.0039  z=+3.60  pos_frac=0.70   FAIL ⚠️
```

Compare to pre-fix (same dataset, raw scipy extrapolation + Wilson anchor):

```
1D:  MAP=0.7279  bias=−0.0021  σ_boot=0.0033  z=−0.64  pos_frac=0.64
2D:  MAP=0.7512  bias=+0.0212  σ_boot=0.0006  z=+37.08 pos_frac=0.64
```

**Key observations:**

- σ_boot in 2D widened by ≈6.5× post-fix (0.0006 → 0.0039), as predicted:
  the unphysically-tight pre-fix σ_boot was a *symptom* of the
  discontinuity (boundary-crossings herded bootstrap MAPs to one value).
  Now σ_boot scales physically with N (1473 events).
- 1D and 2D σ_boot are now comparable (0.0046 vs 0.0039) — that's
  expected behaviour for a likelihood with similar effective DOF count.
- 2D bias is ~16× larger than 1D bias (+0.0141 vs +0.0009). 2D adds
  information; bias should DECREASE, not increase. **This is the bug.**
- pos_frac ≈ 0.70 in both channels: ~70% of per-event MAPs lie above
  truth. Same shared-injection-set signature as before (H4 from the
  earlier handoff); not 2D-specific.

---

## Prime suspect — H3: M_z vs M_source coordinate mismatch

**The mismatch:** the 2D p_det grid is built from injection-campaign
source-frame masses; production queries pass observer-frame redshifted
masses M_z = M_source × (1+z) directly into the grid as if they were
source-frame.

**Code evidence:**

`master_thesis_code/bayesian_inference/simulation_detection_probability.py:407` —
`_build_grid_2d` reads `M_vals = df["M"].values` from the injection CSVs.
`df["M"]` in the injection CSV is the source-frame M of the injected
event (the EMRI's central BH mass *in the rest frame of the source*).

`master_thesis_code/bayesian_inference/bayesian_statistics.py:1304` (numerator,
2D channel):
```python
p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
    d_L, np.full_like(z, _det_M), phi, theta, h=h
)
```
Here `_det_M` is `detection.M` — **the detection's measured BH mass,
which is observer-frame M_z** (LISA measures the redshifted mass; the
"M" attribute of a detection object stores M_z, not M_source).

`master_thesis_code/bayesian_inference/bayesian_statistics.py:1360` (denominator,
2D channel) is internally consistent — both numerator and denominator
pass M_z to p_det. The mismatch is between the *grid* (built from M_source)
and the *queries* (in M_z).

**Comment in code (lines 1298-1300) acknowledges the gap:**
> NOTE: p_det uses the ML mass estimate (detection.M) rather than
> M_gal*(1+z) at trial z. This is a known approximation, not a bug,
> per Phase 14 analysis. The denominator uses M_gal*(1+z) correctly.

The "approximation, not a bug" verdict was made under the post-Tier-3
Phase 45 412-event closure result that PASSED at z=+1.97 — the
approximation didn't matter at that scale. With phase46-merged's 1549
events tightening σ_boot ~6.5× (after the principled-extrapolation fix
restored physical σ scaling), the same approximation now produces a
+3.60σ residual that the 1D channel does not have.

**Quantitative scale estimate:** for typical events at z ≈ 0.5,
M_z ≈ 1.5 · M_source. The grid samples M ∈ [3 × 10⁴, 1 × 10⁶] M⊙
(source-frame, per the test_26 grid_bounds output). Query at observer-frame
M_z lands at ~1.5× higher values. p_det is monotone increasing in M
(heavier → louder for fixed d_L until the inspiral falls out of LISA
band). Net effect: the queried p_det is *biased high* relative to
where the source-frame M actually is. This pulls the 2D L_cat/L_comp
ratio toward higher h-trial values where dist(z_gal, h) places hosts at
slightly different observer-frame distances. Sign and magnitude of the
posterior shift need empirical verification — that's the new session's
job.

**Note:** the M_z appears CORRECTLY in the GW likelihood Gaussian
product on lines 1322-1325 (host source-frame M_gal converted to
M_z_frac coordinates by `host_M * (1+z) / _det_M`). That part is
internally consistent — it's a coordinate transform, not a Jacobian.
**The bug is purely in the p_det grid query**, not in the GW likelihood.

---

## Investigation plan (suggested)

### Step 1 — Verify the mismatch with a direct probe (cheap, ~10 min)

Build a small diagnostic that:
1. Loads the 2D p_det grid at h=0.73 (any h works for the question).
2. For ~100 events from the partial-panel CRB, compares
   `p_det_grid_query(d_L, M_z=detection.M)` vs
   `p_det_grid_query(d_L, M=detection.M / (1+z_event))` where z_event
   is the event's central redshift estimate.
3. Reports the distribution of the difference.

If the difference is small (≲0.01 mean), H3 is not the dominant
mechanism and we move to H3b (entropy mismatch in per-event combine).
If the difference is large (≳0.1), H3 is real and proceed to Step 2.

This avoids any cluster cost. Fits as a follow-up to test_26.

### Step 2 — Decide the fix architecture

Two principled options:

**Option A: Build the grid in M_z space.**
Modify `_build_grid_2d` to compute `M_z = M_source * (1 + z)` for each
injection (using the injection's own z, which is stored in the CSV) and
build the grid axis in M_z. Production queries then use M_z directly,
no conversion needed. The grid samples *match* what's queried.

- Pros: matches the natural coordinate of detection (observer-frame).
  Single source of truth; no per-event conversions.
- Cons: changes the grid; all `posteriors_with_bh_mass/` become stale.
- Status: this is what the inference code already implicitly assumes
  (queries are in M_z). Most-likely-correct fix.

**Option B: Convert query to M_source at evaluation time.**
Keep the grid in source-frame M; modify the query points to
`M_source = M_z / (1 + z_query)`. The numerator integrand uses
`detection.M / (1 + z_event_estimate)`; the denominator uses
the integration variable M directly (already source-frame).

- Pros: smaller diff (no grid rebuild).
- Cons: requires an estimate of z_event for the numerator query.
  detection.z is likely available but if it's noisy, this introduces
  per-event scatter.
- Status: less invasive but introduces a new dependency on z estimate.

**Option C (rejected): "approximate as identity if z << 1."**
This is what the current code does and what Phase 14 documented as "not
a bug". It's not principled at z ≈ 0.3–0.5; rejected.

### Step 3 — Apply the fix, validate, write up

Per the project's principled-physics value (`feedback_principled_physics_choices.md`):
the fix must be derivable from first principles, not chosen to
"reproduce the 1D PASS" or "match the post-fix 1D residual". The
principled choice: query the grid at the same coordinates it was built
at (Option A). If the 2D bias drops to z ≤ 2σ post-Option-A, ship it.

Run `/physics-change` protocol (mandatory for any change to
`bayesian_statistics.py:1304/1360` or `simulation_detection_probability.py`).

Validation: same playbook as today —
1. Property tests: in-grid query identical to before; M_z queries land
   at the right grid coordinate.
2. Re-run the pre-fix h=0.73 closure on phase46-merged (single sbatch,
   ~15 min on cpu_il, $RUN_DIR=postfix_h3_20260506).
3. Re-run h=0.73 closure on Phase 45 412-event sample (preserve prior
   PASS check).
4. Re-run all 4 partial-panel truths (h=0.60, 0.65, 0.70, 0.73) — h=0.60
   was the +55σ outlier pre-fix and showed +bias post-fix; if H3 is the
   driver, h=0.60 should also collapse.

---

## Useful reading order for the new session

| Order | Path | Why |
|---|---|---|
| 1 | This handoff (TL;DR + Prime suspect) | The mechanism in 5 minutes |
| 2 | `.planning/2D-CHANNEL-AUDIT-20260505.md` (Step 1b + Validation) | Today's work + post-fix numbers |
| 3 | `master_thesis_code/bayesian_inference/bayesian_statistics.py:1280-1410` | The 2D numerator + denominator integrand source |
| 4 | `master_thesis_code/bayesian_inference/simulation_detection_probability.py:370-528` (`_build_grid_2d`) | How the grid is built — what the M coordinate actually is |
| 5 | `master_thesis_code/bayesian_inference/simulation_detection_probability.py:700-870` (`detection_probability_with_bh_mass_interpolated`) | The lookup function — what the query coordinates mean |
| 6 | `simulations/injections/injection_h_0p65_task_20.csv` (any one) | First row: confirms `M` column is source-frame, e.g. M=117054, z=0.378 → M_z would be 161298 |
| 7 | `scripts/bias_investigation/outputs/phase46_merged/h0p73_postfix_verdict.json` | The exact post-fix numbers to beat |
| 8 | `docs/H0_BIAS_RESOLUTION.md` §3.14 | Full mechanism + fix narrative |
| 9 | Memory: `feedback_principled_physics_choices.md` | Project's stance on physics modeling |

---

## Things to NOT touch

| File / commit | Why not |
|---|---|
| `master_thesis_code/bayesian_inference/posterior_combination.py` | Tier 3 fix is intact; do not re-introduce outer −N log D |
| Any anchor scheme (Wilson 95% LB, intermediate point estimate, etc.) | Removed today; reintroducing would re-introduce W-LOOP-03 |
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py` saturating face logic | Today's principled bridge handles d_L→0 saturation correctly; H3 is about the M coordinate, not the d_L extrapolation |
| Pre-fix `posteriors{,_with_bh_mass}/` directories on cluster | Kept for diff-against-postfix; do not delete |

---

## Cluster status snapshot (handoff time)

- SSH up via ControlPersist; queue empty (job 4229895 done at ~15:30).
- Cluster HEAD: `2b33cad` ([PHYSICS] principled bridge).
- Cluster workspace: pre-fix `run_closure_h0p73_20260505/` and post-fix
  `run_closure_h0p73_postfix_20260505/` both intact.
- Local prepared CRB (rescaled to h=0.73 on phase46-merged):
  `simulations/cluster_run_closure_h0p73_*/simulations/prepared_cramer_rao_bounds.csv`
  — reusable for the H3-fix validation run.

---

## What success looks like (gate for declaring H3 resolved)

1. **2D z drops to ≤2σ at h=0.73 phase46-merged** (was +3.60 post-bridge).
2. **2D bias ≤ 1D bias at h=0.73** (was 16× larger post-bridge).
   Adding M information must reduce bias, not increase it. This is
   the user's stopping criterion.
3. **Phase 45 412-event h=0.73 closure preserved** at z ≤ 2σ (currently
   z=+1.97 pre-Tier-3-fix and z=+1.97 post-Tier-3-fix in the canonical
   verification record).
4. **No new fitted-to-truth constants** introduced. Any free parameter
   in the H3 fix must be derivable from physics (Value 3 stress-test).

---

## Post-investigation follow-ups (separate sessions)

These were on Step 3/Step 4 of the original 2D-bias plan but are now
secondary to H3:

- **H1 realization-bootstrap (5 seeds at h=0.73)**: if H3 fully resolves
  the 2D residual, this is no longer needed for paper-readiness — but
  still informative for σ_realization vs σ_boot characterization.
- **Full 7-truth panel re-run** (h=0.60 through h=0.85) post-H3-fix:
  the original goal of the multi-truth orchestrator. Once H3 lands,
  re-queue with the full truth set on cpu_il. Note the rsync exit-23
  death pattern in the orchestrator script — fix before re-running.
- **`migrate_crb_to_ecliptic.py` hardening**: still owns a foot-gun on
  post-Phase-36 native-ecliptic CSVs (audit doc references). Not blocking
  H3.

---

## One more thing

The post-fix 1D MAP at h=0.73 is **0.7309** — within 0.001 of truth on
1473 events. That's the cleanest 1D dark-siren H₀ recovery this codebase
has produced to date. The principled bridge fix is shippable for 1D
right now; the only thing blocking paper-readiness is the 2D residual.

Good hunting.
