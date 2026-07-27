# 2D mass-kernel A/B (derivation §3.8 branches a/c, §4 item 4) — readout 2026-07-27

**Purpose.** First discriminator run for the RATIFIED 2D mass-marginal kernel
(`docs/derivations/mass_marginal_2d_kernel.md`, gates M1–M7, implemented
`e9bec6d`): does switching the 2D legs from the linear-Gaussian (+G2d) mass
kernel to the truncated lognormal × R_eff kernel close the measured 2D HIGH
tilt? Scored against the PRE-REGISTERED §3.9 predictions (P2/P3/P4), written
before this run.

**Setup.** Seed-1000 deep venue (inputs symlinked/staged from
`../v1_probe_smeared`: `prepared_cramer_rao_bounds.csv` + `injections/`),
fused 7-point h-grid {0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86}, 3454 events,
seed 1000 in every cell (cluster cells via the new `FORCE_SEED` passthrough,
`bb24b71`), zero zero-likelihood events anywhere. Stack: main @ `e9bec6d`
(post-#50 counted-once PV widths + regenerated catalogue + z-resolved
survival default). Metric: Σᵢ ln pᵢ(h) − Σᵢ ln pᵢ(0.73) from
`simulations/diagnostics/event_likelihoods.csv`.

Cells A′/B ran locally (probe cwd/ recipe); cells A″/B″ ran on bwUniCluster
as per-h 7-task `evaluate.sbatch` arrays (jobs 6061083/6061084, all
COMPLETED, ≈4.5 min/task wall).

## The four cells

| Cell | normalization | z-kernel | M-kernel | 1D MAP | 1D gap(0.86) | 2D MAP | 2D @0.80 | 2D @0.86 |
|---|---|---|---|---|---|---|---|---|
| A′ | absolute_marginal | volume_deconv | gaussian(+G2d) | **0.73** | −69.4 | 0.80 | **+25.6** | +9.2 |
| A″ | absolute_marginal | volume_deconv | **trunc_lognormal** | **0.73** | −69.4 | 0.80 | **+23.8** | +6.7 |
| B | generator_marginal | volume_deconv (flag) | gaussian(+G2d) | **0.73** | −86.1 | 0.80 | **+29.1** | +12.9 |
| B″ | generator_marginal | volume_deconv (flag) | **trunc_lognormal** | **0.73** | −86.1 | 0.80 | **+26.8** | +9.6 |

1D profiles are bit-comparable within each normalization pair (the mass
kernel has no 1D path — the expected no-op, confirmed).

Full 2D profiles (ln vs truth):
- A′: −178.1, −86.7, −26.5, 0, +18.1, +25.6, +9.2
- A″: −173.6, −84.0, −25.5, 0, +17.3, +23.8, +6.7
- B : −204.8, −98.4, −29.5, 0, +20.2, +29.1, +12.9
- B″: −198.6, −94.9, −28.2, 0, +19.1, +26.8, +9.6

## Kernel movement (gaussian → trunc_lognormal, same normalization)

| h | A′→A″ | B→B″ |
|---|---|---|
| 0.76 | −0.8 | −1.1 |
| 0.80 | **−1.8** | **−2.3** |
| 0.86 | −2.5 | −3.3 |

Direction: toward truth at every broadened-kernel h, in BOTH normalizations.
Magnitude: ~2–3 ln of the ~25–29 ln excess; 2D MAP unmoved at 0.80.

## Scoring against the pre-registered §3.9 partition

- **P2 (direction): CONFIRMED.** The kernel moves the 2D profile DOWN
  (toward truth) in every broadened-kernel cell.
- **P3 (magnitude): the TOY-LEVERAGE branch, at its low end.** Recovery
  ≈ 2–3 ln of the pre-registered "roughly 2–13" band; residual 2D
  MAP = 0.80 ≥ 0.78. The operative mass-kernel leverage in production is
  the host-σ_eff scale (the toy's axis), NOT the GW-window scale — the
  §3.8 "top open uncertainty" (leverage mapping) is hereby RESOLVED to the
  toy side. Not the middle band; not the sufficiency branch.
- **P4(ii): FIRES.** Post-fix 2D MAP > 0.76 with 1D at truth in the
  absolute_marginal cells ⇒ **necessary-but-not-sufficient is now
  measured**: per the pre-registration, branches (b) CRB proj/cross-
  covariance channel, (d) selection M-leg (FIX-3 Σ_glob_wbh composition,
  ship-together rule), and (e) 2D numerator quadrature (n=50 vs 200)
  escalate to first priority.
- **P4(i)/(iii): do not fire** (no upward movement; no interior-host
  anomaly surfaced by the goldens — the parity suite passed with only the
  two reviewed σ_lnM = 0.1 boundary-case updates).

## Additional findings

1. **Branch (c) (cell-B instrumentation mismatch) is largely CLEARED as the
   residual's owner**: the same ~+24–29 ln 2D tilt appears in
   `absolute_marginal` (a real candidate mode, no generator legs), and the
   kernel movement is normalization-independent (−1.8/−2.3 ln). The
   residual is a property of the broadened-z 2D channel itself, not of the
   B-cell's mixed legs.
2. **The old cell-A 1D rail is GONE at the current stack**: old cell A
   (49b9ade-era) railed 1D at 0.86 (+54.2); A′ now peaks at truth with a
   monotone fall-off (−69.4 at 0.86 — matching the z-resolved survival
   probe's pre-registered −69 ln figure,
   `../zres_probe_20260726/PROBE_RESULTS.md`). The ratified real-data 1D
   combination (absolute_marginal × volume_deconv) is healthy on this deep
   venue at the current stack. (Consistent with, and extending, the zres
   probe; the #50 width changes left cell B essentially unchanged:
   +29.1 now vs +29.4 pre-#50.)
3. **Cost (§4 item 7): resolved empirically.** The trunc_lognormal cells
   ran at gaussian-parity speed (≈4.5 min per h-task incl. ~2 min setup,
   16-CPU nodes; CPU-seconds dominated by setup, not the GH/GL legs). The
   ≤×5 worst-case estimate is far above the realized cost; no optimization
   needed.

## Consequences for the derivation ledger

- RATIFY-M6 position unchanged and now evidence-backed: the (M1) kernel is
  in production-quality shape (adopted, cheap, correct limits) but the 2D
  channel REMAINS OPEN. (Round-2 update below: branches (b)/(e) measured
  NULL the same day; the residual attributes to branch (d) by
  elimination.)
- The §3.5 mechanism (mz(z) as a second z-likelihood) predicts the
  kernel-shape term specifically; its measured share (~2–3 ln of ~26)
  BOUNDS the shape sub-effect. Round 2 then killed the μ_cond(z)-sweep
  and quadrature explanations — the remaining tilt enters through the
  selection legs (branch d).

## Round 2 (same day): branch (b) and (e) discriminators — BOTH NULL

Diagnostic env overrides (`fe0ca3e`, default-off, byte-identical goldens):
`MTC_ABLATE_MZ_PROJ=1` (2D conditioning → the consistent independent-M_z
marginal pair: proj = 0 AND σ²_cond = Σ₄₄) and `MTC_HOST_QUAD_N=200`
(stack-wide z-quadrature order). Cluster jobs 6061850/6061851/6061852
(ablated A′, ablated A″, n=200 A′; all COMPLETED, override warnings
verified in the .err logs, git_commit fe0ca3e in run_metadata).

| Diagnostic | vs baseline | max per-event \|Δln L\| (2D) | total Δln @0.80 |
|---|---|---|---|
| A′ proj-ablated | A′ | 4.4×10⁻⁷ | +0.000 |
| A″ proj-ablated | A″ | 3.2×10⁻⁶ | +0.000 |
| A′ n=200 | A′ | 2.9×10⁻¹⁰ | +0.000 |

- **Branch (b) — CRB proj/cross-covariance channel: NULL.** Dropping the
  d_L–M_z cross-covariance is numerically a no-op at the 10⁻⁶-ln-per-event
  level: on this venue the CRB conditional is already ≈ the marginal
  (negligible d_L–M_z correlation in the Fisher matrices). The §3.5
  "μ_cond(z) sweep" mechanism cannot carry the residual.
- **Branch (e) — 2D numerator quadrature: NULL.** n = 50 → 200 is
  converged to 10⁻¹⁰ ln; no aliasing through the mz(z) modulation.

**Attribution by elimination.** With (a) measured at ~2–3 ln, (b)/(e) null,
(c) cleared by the A-cell reproduction, and (f) bounded at +0.004–0.006
per-event-sum scale by the P–P harness, the ~+25 ln 2D HIGH residual falls
to **branch (d): the selection-side M-composition** — Σ_glob_wbh /
2D-selection legs (point-evaluated in M, pooled survival) — i.e. the FIX-3
§7.1 thread (`DERIVATION_ZRESOLVED_SURVIVAL.md`), whose z×M_z-resolved
Σ_glob variant already has a measured −58-ln-scale predicted effect — the
right order of magnitude for the first time. The FIX-2/FIX-3 ship-together
gate is now the critical path for closing the 2D channel.

## Provenance

- Cells A′/B: local probe runs (this dir, `run_cells.sh` + follow-up B
  driver), seed 1000, `run_metadata.json` per cell.
- Cells A″/B″: bwUniCluster jobs 6061083/6061084 (7-task arrays,
  `--array=0,5,15,21,27,34,40` on the 41-value grid = the 7-point fused
  grid), `run_20260727_massab_cell{App,Bpp}` in the emri workspace,
  retrieved here; per-task `run_metadata_*.json` (seed 1000,
  host_mass_kernel recorded).
- Stack: `e9bec6d` (kernel + flag), `bb24b71` (sbatch passthrough).
- Probe convention: `simulations/`, logs, and metadata are local-only;
  this readout is the committed record.
