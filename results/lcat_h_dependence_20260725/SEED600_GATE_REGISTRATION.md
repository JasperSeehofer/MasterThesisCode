# Pre-registered gate: seed600 shallow must-not-change (third arm)

**Registered 2026-07-26, BEFORE the third arm runs.** Venue: `run_20260628_seed600`
(3,355 events, shallow z≤0.5 pool via `--allow_low_pdet_coverage`; Ω_m era mismatch —
this venue supports RELATIVE A/B statements only, never absolute closure claims).

## Measured baselines (jobs 6043672/6043673, code @ c87caba)

| Arm | 1D MAP | 2D MAP | n_used |
|---|---|---|---|
| volume_deconv (production default) | 0.745 | 0.755 | 3353/3355 |
| absolute_marginal (V1) | 0.775 (+0.030) | **0.86 RAIL** | 3353/3355 |

V1-alone **fails** the shallow gate (n̄_w calibration; the with-BH channel's
mass-composition violation — the exact defect FIX-3 removes). Recorded, expected
in hindsight, and moot for the production candidate.

## Gate for the third arm (generator_marginal + --pdet_z_resolved)

Pass requires ALL of, per channel, relative to the **volume_deconv arm**:
1. |ΔMAP| ≤ max(0.010, 2·σ_boot^vdeconv) — σ_boot per §3.17 bootstrap methodology
   (~0.006 on this venue → effective tolerance 0.012);
2. MAP strictly interior (no grid-edge rail);
3. n_used identical (3353/3355);
4. no new zero-likelihood events.

Rationale: on a shallow, p_det≈1, catalogue-dominated venue the new estimator
must reproduce the previously validated estimator's inference within its own
statistical resolution. The old p_det→1 *algebraic identity* argument covers only
the normalization tier; the point-N_g leg (point/point pairing) changes the
numerator sharpness, so agreement is an empirical gate, not an identity — hence
this registration. A failure blocks production adoption pending diagnosis
(first suspect per derivation risk 4: low-z events with weak candidates shifting
weight to B_num under the sharper numerator).

Verdict to be appended below by the session that reads out the third arm —
after this file is committed, no edits above this line.
