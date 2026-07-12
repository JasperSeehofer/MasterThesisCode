# Next-session entry — Part 1 `volume_trunc` is DONE and FALSIFIED

**This supersedes `.planning/HANDOFF-VOLUME-TRUNC-EXEC-20260712.md` (that kickoff is executed).**

## What happened (2026-07-12, commit `c4a1c7d`)

Part 1 of the production host-z kernel fix (`volume_trunc` = unified per-host numerator
support over `[z_g−4σ, z_g+4σ]` + z-floor 0) was implemented faithfully behind an isolated
`normalization_mode` and run through its **decisive seed600 494-event A/B gate**. It was
**empirically FALSIFIED**: it worsens the shallow venue (1D mean **0.745 → 0.800**, MAP
0.73 → 0.80, posterior collapses onto h=0.80). The `volume_deconv` arm reproduced the
established reference exactly, so the result is the kernel, not the harness.

**Mechanism** (`results/volume_trunc_ab_20260712/FINDING.md`, `quadrature_diagnostic.py`):
1. **Quadrature aliasing (dominant):** `fixed_quad(n=50)` — correct for the narrow GW window —
   is numerically invalid over the WIDE host window; the sparse GL nodes miss the narrow GW
   peak (n=50 → 0.0 vs exact 0.24–0.65), h-dependently → collapse onto the aliasing-favoured h.
2. **Genuine high-h tilt:** even the exact host-window numerator increases with h in the shallow
   regime.

⇒ The numerator-window unification is **NOT the +0.013 lever** and is numerically broken as
specified. Do NOT deploy. `volume_trunc` is retained as EXPERIMENTAL/FALSIFIED (not CLI-wired);
`volume_deconv` stays the golden default (byte-identical, untouched).

## State of the code (all committed, `physics/zero-host-completion-fallback`)

- `volume_trunc` scalar + batched kernels; `volume_deconv`/`local_ratio` **byte-identical**
  (golden regen additions-only, batch≡scalar bit-identical, full CPU suite **889 passed**).
- Tests: volume_trunc pins + σ_z→0 limiting case + prior-shape h-independence
  (`test_bayesian_statistics_host_z_kernel.py`, `test_kernel_parity.py`).
- Driver `scripts/volume_trunc_ab.py`; finding + reproducible diagnostic in
  `results/volume_trunc_ab_20260712/`.

## Next direction (USER-GATED — needs its own /physics-change)

The shallow +0.0132 attribution stands ([L8]: σ_z/z-at-low-z truncated-volume-kernel Eddington
effect), but its cure is NOT the numerator window. Open the scoping toward **Candidate B**
(photo-z-marginalized SOFT membership, scoping §3) co-designed with the **[L7] distance-error
coupling** — with the NEW hard constraint that ANY wide-window numerator integral must use a
**peak-aware / adaptive / high-order** quadrature (the narrow GW peak must be resolved). That is
a larger change than Part 1 scoped. **User's call**: pursue a quadrature-robust reimplementation
of the numerator window, or abandon the numerator-window idea and go straight to Candidate B.
Decisions recorded in `.planning/DECISIONS-20260712.md` §PROD-Part-1-OUTCOME.

## Everything else unchanged

Cluster-gated items (D2 combined deploy, EXP-40, campaign, #29/#30) still await cluster return
per `.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md` L-F. Paper A on hold (D3). 2D +0.025
residual deferred to campaign (D4).
