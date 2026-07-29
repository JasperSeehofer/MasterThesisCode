# Campaign #51 — idealized-host baseline readout (2026-07-29)

**What this is.** The first end-to-end demonstration that the generator and the
evaluation agree in this pipeline: injected h is recovered, on a corrected-
physics stack, at two independent seeds. **What it is NOT:** a forecast. The
host redshifts and masses are used point-to-point (no measurement noise
realized), so recovery-on-truth is guaranteed by construction — see
`idealization_audit/IDEALIZATION_LEDGER.md` and the caveat section below.

## Stack

Corrected-physics stack `a9f29e8`+ (this session's `[PHYSICS]` fixes):
`ecb56d6` single-source mass boundary (Babak band [1e4, 1e7], all secondary
clamps removed) · `49251f3` confusion-noise TDI transfer (the pre-fix PSD made
the detector ~10³× deaf below ~1 mHz) · `e419062` plunge-window initial
conditions + official T_mission = 4.5 yr (Colpi et al. 2024) · `a9f29e8`
separatrix-sign skip · `acaa0af` provenance columns · `ec09ed0` measured-mass
domain (removed a hardcoded 1e6 clip that had pinned 8–9 % of events).
Selection: `injection_pool_mix200k_20260728` (200,100 rows, stratified
mix3_50_25_25) — all pre-registered acceptance criteria PASS
(`acceptance/ACCEPTANCE_REPORT.md`).

## Results (resolved on a 1e-4 zoom grid)

| seed | h_true | N events | MAP | mean | σ_h | 68 % CI | bias |
|---|---|---|---|---|---|---|---|
| 61000 | 0.73 | 1588 | 0.72990 | 0.72993 | 0.00030 | [0.72962, 0.73023] | −0.24σ |
| 62000 | 0.73 | 1542 | 0.72990 | 0.72986 | 0.00039 | [0.72946, 0.73026] | −0.36σ |
| 64000 | 0.67 | — | — | — | — | — | PENDING (closure test) |

Both fiducial seeds recover truth inside the 68 % interval with symmetric,
Gaussian peaks (ln L falls to −123/−115 at ±0.005, i.e. ±15σ, matching the
Gaussian prediction to a few percent). Both biases are NEGATIVE at ≈ −0.3σ;
with two seeds this is not significant, but the sign coincidence is worth
re-testing with the realistic-run pull statistics.

The peak is ~15× narrower than the production h-grid's 0.005 peak step, which
is why the un-zoomed posterior looked like a delta spike; the
`H_VALUES_OVERRIDE` zoom hook (`0818ced`) exists for exactly this.

## Where the information comes from (MEASURED)

100 % of the constraint is carried by the **76 in-catalogue events (4.8 %)**;
the 3 loudest (SNR 995–1425, z ≈ 0.016–0.021) carry 46 % alone. The other
~1510 detections are dark hosts contributing ~1 % of the curvature. The
per-event budget matches "z exact + GW d_L error only" to ~5 %:
σ_H0/H0 ≈ 0.38 % / √76 = 0.044 %, i.e. exactly the measured width.

## Caveat — why this is a consistency test, not a measurement

All 76 information-carrying hosts are GLADE+ **photometric**-redshift
galaxies with median σ_z/z = 49 %, whose redshift is injected as truth and
point-evaluated by the production δ-kernel. With the host redshift exactly
right, each golden event pins H0 at its GW distance precision. Realistic
counterfactuals on the SAME events: 0.22–0.30 (perfect spec-z + PV) to ≈ 3.6
km/s/Mpc (catalogue photo-z widths). The campaign-#53 realistic run
(`docs/derivations/realistic_host_observation_model.md`, RATIFIED) replaces
the point/point pairing with realized noise; its forecast is
σ_H0 ≈ 1.3–1.7 km/s/Mpc.

**Quote this baseline only as "generator↔evaluation consistency at the
0.0003-in-h level", never as a LISA H0 forecast.**

## Provenance

Cluster run dirs `$WS/run_20260729_seed{61000,62000}` (+ `/zoom`); local
mirrors `run_seed61000/`, `run_seed62000/` (posteriors + zoom; the prepared
CRB CSVs and the 200k pool are gitignored bulk — canonical copies on the
workspace). Pilot #3 narrowing decision: `PILOT3_READOUT.md`
(no narrowing — detections to detector m = 6.96, full Babak band FINAL).
Pilot #1 (`PILOT_READOUT.md`) is QUARANTINED (pre-PSD-fix physics).
