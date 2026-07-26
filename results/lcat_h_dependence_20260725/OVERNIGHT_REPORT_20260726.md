# Overnight autonomy report — 2026-07-25/26

Goal (user directive): find the scientifically sound Bayesian inference with no residual
bias; follow breadcrumbs, step back periodically, no truth-tuning.

## Where the science stands (one paragraph)

The deep-venue rail had TWO stacked causes. Cause 1 (LOW rail, h=0.60): host
misassociation — impostor-only candidate balls dominating a self-normalized L_cat. The
**absolute-mass marginal (V1)** fixes it, verifiably: per-event slopes −12.1 → +0.74 mean,
all 12 instrumented events flatten, the mechanism metrics pass. Cause 2 (HIGH rail, h=0.86,
exposed by fixing cause 1): the catalogue↔population calibration `n̄_w = Σ_glob/β_G` and
the selection normalization rest on the Option-A constant-comoving-density identity, which
the real catalogue **measurably violates** (detected-z composition ×8.4 off at z<0.05,
×0.6–0.67 at z=0.1–0.25 — E1 §FIX-3), leaving a shared ~+0.3/h tilt that rails 3454 events
across a 0.26-wide grid even though each event is nearly flat. The σ_z-smearing of Σ_glob
(landed, opt-in) is principled but removes only ~20% of the residual (measured
+0.368→+0.435 vs pure-h³ target ≈ +0.74 relative); the remainder is composition, not
kernel asymmetry. Separately, E1 proved the completion term `B_num/D` is NOT defective:
the 0.612 fallback-subset peak is a subset-conditioning artifact (membership is
h-informative), the generator self-consistency MC closes at truth under the correctly
conditioned statistic, and real-vs-MC agreement reaches 4 decimals on clean subsets.

## What landed (all on branch `physics/absolute-mass-marginal`, pushed)

| Commit | What | Status |
|---|---|---|
| `49b9ade` | `[PHYSICS]` `absolute_marginal` mode (V1, user-approved pre-implementation) | tests 976 green; probe run |
| `f9c58f4` | `[PHYSICS]` opt-in `--smear_global_selection` (σ_z symmetry, R4) | tests 982 green; probe run; **needs morning ratification** (implemented under overnight-autonomy grant, non-default) |

Probes (seed1000, 3454 events, 7-point grid, both channels):
| Config | 1D MAP | 2D MAP | 0.73→0.86 gap (ln, 1D) |
|---|---|---|---|
| volume_deconv (EXP-40, 41-pt) | 0.60 RAIL | 0.60 RAIL | (railed LOW) |
| absolute_marginal | 0.86 RAIL | 0.86 RAIL | +54.2 |
| absolute_marginal + smeared Σ_glob | 0.86 RAIL | 0.86 RAIL | +50.0 |

Also: P–P harness gained `mixture_mode="absolute"` (worktree branch
`worktree-agent-a8b42ff5c1f0382f4` @ `7c513dd`, 937 green) — honest null: the harness's
single-candidate universes cannot express impostor balls, so it cannot test V1; needs a
multi-galaxy-catalogue extension to become the right instrument.

## Decisions for the author (ranked; each is a physics-change gate)

1. **FIX-3 — generator-consistent selection normalization** (E1; the root of the HIGH
   rail). The Option-A identity `Σ_cat w_g P_det ≈ n̄_w β_G` is violated by the real
   catalogue's z-composition. Candidate form: `D_gen = F·Σ_glob + (1−F)·β_Ḡ` (E1
   measured `d log D_gen/dh ≈ −1.02` vs current −1.52), but the correct final form needs
   a real derivation — it changes the master normalization and moves the host and fallback
   channels in opposite directions; full-mixture evaluation is the only valid readout.
   THIS is the main event. Estimated effect: the remaining ~+0.3/h shared tilt is
   composition-sourced; a consistent D/n̄_w treatment targets exactly it.
2. **FIX-2 — z-resolved survival `S_z(d_L)`** in D/β_Ḡ/β_G (E1): the SNR∝1/d_L horizon
   trick is exact only at fixed z. Predicted +0.02…+0.05 on completion-channel statistics.
   Complements (not substitutes) FIX-3.
3. **Ratify or revert `f9c58f4`** (σ_z-smeared Σ_glob): principled symmetry, measured
   small (+0.067/h of the +0.38 target). Recommend: ratify as opt-in, fold into FIX-3's
   derivation (whatever replaces n̄_w must keep kernel symmetry).
4. **Gate definitions** (no physics): C1 must be membership-conditioned
   (`Σlog p_i − N·log P_fb(h)`, E1 FIX-1) — the naive fallback-only closure I posed
   overnight is not a theorem and would fail for a perfect estimator.

## Corrections to my own overnight framing (step-back honesty)

- I posed sub-channel gate C1 wrongly; E1's MC disproved the premise. Kept for the record.
- The n̄_w-residual-dominates hypothesis after the first V1 probe was ~half right: the
  smearing experiment measured its σ_z component at only ~20%; the composition component
  (Option-A violation) is the real carrier.
- The P–P "both modes miscalibrated" deep-cell result I flagged as a completion-term
  suspect is better read as the harness's own subset-conditioning artifact (same class as
  the C1 error) — pending the harness upgrade, it is NOT evidence about production.

## Blocked on cluster 2FA (queued)

seed600 shallow A/B (must-not-change gate for V1); multi-seed deep evals (900/2000/3000/
90000) for the final residual-bias measurement; full 41-grid confirmations. One `! ssh
bwunicluster echo ok` re-arms everything.

## Artifacts index

- `DERIVATION_ESTIMATOR_REDESIGN.md` — V1/V2 derivation (V2 rejected: not a likelihood)
- `D1_EMPIRICAL_DECOMPOSITION.md`, `D2_STRUCTURAL_AUDIT.md` — mechanism identification
- `completion_bias/E1_COMPLETION_BIAS.md` — completion-term exoneration + FIX-1/2/3
- `v1_probe/`, `v1_probe_smeared/` — probe working dirs + logs (per-h sums in probe.log)
- `results/pp_coverage_absolute_20260726/` — harness null + fidelity caveat
