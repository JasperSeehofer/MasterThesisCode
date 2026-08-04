# Gate B — adversarial refutation of GFRAC_DERIVATION_PACKAGE.md (2026-08-05, overnight session)

Independent opus-tier refuter; all numeric checks re-run from the diagnostics CSVs
and the shipped code. Verdicts per load-bearing claim:

| # | claim | verdict | key evidence |
|---|---|---|---|
| 1 | Measure argument (x_M measure numerator-only; α(h) pure number; completion-leg measure invariance exact) | **SURVIVES** | Re-derived from MFG; conditional-centre symmetry vs `bayesian_statistics.py:1990-1992`; factorisation precondition holds (`:3241`,`:3258`,`:4044-4045`); h-variation of the M_z^obs normalisation measured at 7e-9. Caveat: conditional on numerator carrying no selection factor (see N-2). |
| 2 | 0.1% slope closure, s_dex = −0.43 | **SURVIVES + AMENDED** | Reproduced: Σ 243.53 vs 243.95 predicted; not circular (s_dex analytic from `emri_rate.py:96`,`:235-261`). AMENDMENT: φ is a **broken** power law — `kappa_cap` kink at M=10⁵ (`emri_rate.py:169`,`:198`), s_dex flips −0.43→+0.07 below; event 953 straddles it (98% slope deviation, g_frac turns over at h≈0.733). P1 as written FAILS today (40/1588 events > 3e-3). |
| 3 | Exact 1D-marginal identity ∫(2D compl. num.) dM_z^obs = 1D compl. num. | **SURVIVES (verified independently)** | 100 events × 3 h: 1 + 7.66e-6 (trapezoid-converging), h-variation 7e-9. P3 pin should be ≤1e-5 (code's own ∫φ dM = 1+7.7e-7 leaves no 1e-6 headroom). |
| 4 | Decomposition "0.600 rail + 0.060 + 0.120 genuine tilt" and the "re-attribute to 1D-rail bias" consequence | **REFUTED** | Violates the standing never-add-MAP-displacements rule (`GATE_PACKAGE_FINAL.md:609`, `DERIVATION_C7_HOSTZ_KERNEL.md:545-547`); frozen-2D likelihood flat over [0.63,0.745] (argmax beats 0.640 by 0.021 nats); identical tilt yields 0.120 (iiib) vs 0.160 (joint) — not transferable; ln p^2D = ln p^1D + ln g + const fails by 33.8/23.5 nats; curvature figures not reproduced (sign change found). What SURVIVES: (lnL2_live − lnL2_frozen) = Σ Δln g_frac to ≤0.8 of 63.5 nats — the freeze is surgically clean and the tilt real; only its MAP-space decomposition is unsupported. |
| 5 | Flat-φ limiting case (slope flips negative) | **SURVIVES** | Toy monkeypatch: g·(1+z)/M_z,det constant to 4e-16; dln g/dln(1+z) = −1 exactly; real-φ above break +0.43000 exactly. |
| N-1 | Gate (i) near-vacuous (2D catalogue leg dead for most events) | **SURVIVES exactly** | iiib 0.8149 identically-zero / mean share 0.0543 / median 0.000; joint 0.6178 / 0.0573 / 0.000. |
| N-2 | 1D-side selection-marginal defect candidate | **REAL in structure, MIS-SCOPED** | Detection model is θ-deterministic (`simulation_detection_probability.py:175-179`), so the correct hierarchical numerator carries p_det(θ) in **both** channels — as S_4D inside g_i's M-integral (2D) vs S̄_φ (1D); it does NOT cancel from g_frac. The 2D channel is also reduced data (11 of 14 params discarded). §6.4's "D1 cannot reach g" is CONDITIONAL on N-2's resolution. |
| §6.5 | Population cross-check (implemented tilt 8% too SMALL — adverse) | **SURVIVES** | Independently: event-weighted s_dex −0.4251 (pkg −0.4253); true-population tilt 263.1 vs 244.0 → −7.3% (pkg −8.0%). |

## Overall: goes to the author WITH AMENDMENTS (not rework)

Required amendments (applied in the package by the follow-up edit):
1. Rewrite §7 — drop the additive decomposition and "0.13 too low"; state the defensible version (frozen-2D flat over [0.63,0.745]; tilt = live−frozen exactly; MAP image venue-dependent and not decomposable).
2. Fix §5(a)/§6.5 — φ is a broken power law (kappa_cap kink at 1e5); internal kink IS active (event 953); amend §6.4 accordingly.
3. Repair P1 (kink-aware); relax P3 to ≤1e-5.
4. Downgrade §6.4's D1 exclusion to conditional on N-2.
5. R-A's consequence clause ("re-attribute to the 1D rail") is NOT established by this package — the pre-registered §9 closed-loop synthetic-universe test is the deciding measurement. R-B stands (N-1 reproduced; algebraic proof strictly stronger than gate (i)). R-C re-scoped: not a 1D-only question.

## Incomplete checks
- Posterior-weighted-z vs plug-in z* in the closure: UNDETERMINED (f_k weight unreachable outside the estimator; the systematic +0.1% median offset is in the Jensen direction, indirect evidence the plug-in is adequate).
- The §9/G4b synthetic-universe run itself — the only measurement that decides R-A — was not run (it is tonight's main thread).
