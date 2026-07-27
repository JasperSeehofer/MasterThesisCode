# FIX-3 §7.1 joint-grid A/B — readout 2026-07-27 (pre-registered scoring)

**Setup.** Cells A‴/B‴/grid-only = A″/B″ + `--pdet_wbh_z_resolved` (jobs
6065823/6065824/6065825, all COMPLETED, ~5–6 min/task incl. first joint-grid
builds; `MTC_WBH_GRID_ONLY=1` on the control, override verified in .err;
`pdet_wbh_z_resolved: true` in run_metadata). Same seed-1000 venue, seed
1000, stack `608426b`. Scored against `fix3_zmz_catalog_selection.md` §3.9
(z3-refined 2026-07-27, committed BEFORE these runs at `29798b8`).

## Profiles (2D, ln vs truth) and flag deltas

| cell | 0.60 | 0.65 | 0.70 | 0.76 | 0.80 | 0.86 | MAP |
|---|---|---|---|---|---|---|---|
| A″ (baseline) | −173.6 | −84.0 | −25.5 | +17.3 | +23.8 | +6.7 | 0.80 |
| A‴ (joint) | −173.8 | −84.0 | −25.4 | +17.1 | **+23.3** | +5.6 | 0.80 |
| grid-only | −174.7 | −84.7 | −25.7 | +17.5 | +24.4 | +7.8 | 0.80 |
| B″ (baseline) | −198.6 | −94.9 | −28.2 | +19.1 | +26.8 | +9.6 | 0.80 |
| B‴ (joint) | −203.0 | −96.8 | −28.7 | +19.4 | +27.3 | +10.1 | 0.80 |

Flag deltas: A″→A‴ @0.80 = **−0.51**, @0.86 = −1.18. Grid-only @0.80 =
+0.56, @0.86 = +1.01 ⇒ conditioning-only (A‴ − grid-only) @0.80 = −1.07,
0.73→0.86 increment −2.19. The grid/interpolant confound is the SAME order
as the conditioning signal, opposite sign — the §4 item-12 control earned
its place.

## Gate scoring

- **P2 (increment −6.5 ± 4, i.e. [−10.5, −2.5]): FAILS LOW.** Measured
  0.73→0.86 increment: raw −1.18; grid-corrected −2.19 (just outside the
  band). The z3 table arithmetic over-predicted the production movement by
  ×3–5.
- **P3 A-cell (Δ@0.80 ∈ [−9, −3]): NULL band** (−3 < Δ < +3): raw −0.51,
  grid-corrected −1.07.
- **P4 no-ops: ALL PASS.** A-cell 1D **bit-identical** (verified
  array-equal). B-cell 2D−1D residual **exactly invariant** (+57.16 →
  +57.16) — the Z5 atomic-switch ledger is confirmed at machine precision.
  B-cell 1D moved (not bit-identical ⇒ the flag DID reach D_gen), but by
  +0.45 @0.86 / −4.37 @0.60 — NOT the predicted −6.5-scale monotone shift:
  the quantitative D_gen-shift prediction fails alongside P2 (the
  qualitative falsifier does not fire).
- **P5: NULL.** Per the pre-registered partition: the production-axis
  (d1) increment is genuinely small; the (d1)-sufficiency reading dies;
  branch **(d2)** (selection-side M scatter/truncation, RATIFY-M5
  deferral) and **(g1)** (81 % of catalogue weight on the m_max clamp)
  are the enumerated owners of the remaining ≈ +23 ln 2D residual. The
  estimator derivation itself stands (all structural tests green).

## The mandated follow-up (pre-registered in P5's consistency clause)

z3's TABLE movement was material (−6.5 shipped) while the A/B is null ⇒
**probe/production parity is broken** and must be audited before any
physics conclusion about (d1)'s true size: candidates, in order — (i) the
binned catalogue profile vs production's row-by-row Σ_glob_wbh sum (the
registered 0.589-vs-0.556 parity item; the 3D channel's parity is 4×10⁻⁵,
the 2D channel's was never established); (ii) the §3.3-C convention deltas
(probe step-in-d_L / linear-in-m vs shipped linear-in-d_L / lifted-knot
erf-sum); (iii) per-cell shrinkage weights vs the probe's uniform-w̄
approximation on the actual query set; (iv) the m-clamp serving 81 % of
the weight identically in both arms (clamped queries are insensitive to
the u-axis by construction — plausibly THE parity breaker: if the clamp
dominates, most catalogue queries never feel the conditioning).

> **[SUPERSEDED 2026-07-28 — P1 audit executed]** The follow-up above ran:
> `P1_PARITY_AUDIT.md` (this dir). Hypothesis (iv) REFUTED (clamped queries
> carry ~75–90 % of the conditioning movement); the ×3–5 shortfall was a
> probe axis-translation error + 6 % baseline value error; the −6.5±4 gate
> is RETIRED and consequence 2 below (clamp-suppression narrative + the
> "(d1) at full size post-campaign" prediction) is WITHDRAWN — the
> replacement pre-registered prediction is in the audit §6 and
> `docs/campaign_redesign_51_design.md` §6.

## Consequences

1. **(d1) measured ≈ null in production.** The 2D residual's owner
   expectation shifts almost entirely to **(d2) + (g1)** — both of which
   the ratified campaign redesign (#51: mass bounds to the scientifically
   correct limit + ESS-floor pool) directly attacks. The campaign is now
   unambiguously the 2D critical path.
2. Hypothesis (iv) links (d1)-null to (g1) mechanistically: with 81 % of
   catalogue weight on the mass clamp, the joint conditioning acts on only
   ~19 % of the weight — consistent with the measured ×3–5 shortfall vs
   the table (which the probe evaluated including clamped cells at their
   node values). If the parity audit confirms this, (d1) is not wrong but
   SUPPRESSED by (g1), and the new campaign (which removes the clamp)
   should reveal it at full size — a pre-registrable prediction for the
   post-campaign rerun.
3. The flag stays default OFF (it was never to be promoted outside the
   joint ship gate); its implementation is validated by the P4 exact
   no-ops and stands ready for the post-campaign universe.

## Provenance

Jobs 6065823–25 (7-task arrays, seed 1000, venue = v1_probe_smeared
inputs); baselines cellApp/cellBpp (this dir, same day, same seed);
diagnostics retrieved to zmzApp/zmzBpp/zmzGridOnly; stack `608426b`
(implementation), `29798b8` (z3 pre-registration), `7219cd9` (ratified
derivation rev. B).
