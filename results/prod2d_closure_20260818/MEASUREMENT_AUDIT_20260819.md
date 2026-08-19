# Independent measurement audit — rows #119–#128 — 2026-08-19

**Provenance:** fresh-context top-tier agent, author-mandated (row #129 item 2: "double
checks the measurements … if the numbers motivate the decisions"). All recomputations
independent from banked CSVs with fresh scripts; report verbatim in substance.

## Findings

1. **T0 — OK.** mean_h 0.784222/0.796707, σ_boot 0.01136/0.01199, z 4.77/5.56,
   jackknife-889 ratios 1.199/1.0005 → ROBUST; 1D 0.6040/0.6074 low-edge rail confirmed;
   physics-floor no-op confirmed; scorer implements P7-2 as registered incl. N-0 hard-stop.
   Nit: joint_r1 anchor rounds to 0.7967 (diff 1.07e-4, inside the 5e-4 gate, disclosed).
2. **Regression — OK.** S1/S2/S1a/S3 reproduce exactly; c_e′ §2b identity verified from raw
   columns (0 violations); S2 orientation genuinely anti-M-B; decision-table walk lands
   R-MIXED under either reading of rule 3; stage-2 UNDERPOWERED-NULL verified (n=76).
   Minor: S4 OLS standardization inconsistency (non-adjudicating); the
   "impostor-overlap" interpretive sentence was subsequently undercut by the counterfactual
   (correctly anticipated by P8 in the VERDICT itself).
3. **Production counterfactual — OK.** ΔV1 recomputed +0.000957/+0.003165 (quoted
   +0.0010/+0.0032); N-0 verified as a matched-row comparison (1588/1588 joined, 0.0 /
   6.1e-14); ΔV2 ladder reproduces; C-MIXED correct by the registered letter;
   ownership-refuted-at-materiality reading numerically supported. Nit: N-2 scored at
   h=0.72 only vs "at the probe h-values" (immaterial at 93.5%/84.7% vs 10%).
4. **Row #119 fusion counterfactual — OK.** fused−off 2D mean_h = +0.000396/−0.000532;
   "near-inert in production 2D" fully supported.
5. **P7-4 budget / s_Edd — CAUTION (load-bearing).** Sign convention CORRECT
   (r = Δ − (−0.020) is the right direction; arithmetic verified). **But the −0.020 input
   is STALE:** `docs/gates/G7row9_N5_postDgfix_SUMMARY.md` (2026-07-12) records the
   post-D_g-fix Eddington-in-M impact as **−0.0022** and flags the code comment for
   refresh; the −0.020 came from a pre-`713fbd1` artifact with a pathologically railed 2D
   posterior (edge_mass 0.216), measured with the log-linear tilt transformation whose sign
   the current code's own docstring (bayesian_statistics.py:606-609) disavows at GLADE
   σ_rel ≈ 1, at seed600/494-events/7-pt — not the current code path, venue, or grid.
   Neither the closure prereg, verifier Part VII, nor row #127 caught the standing flag.
   **B-UNOWNED is robust to any plausible s_Edd ∈ [−0.02, +0.02]** (r ≥ +0.034/+0.047 ≥
   2σ_total), but the headline residual should be corrected to **≈ +0.056/+0.069
   (~2.3–2.9 σ_total)** or carried with an s_Edd band; the :5530-region comment should be
   refreshed; ideally s_Edd re-measured production-natively (current exact-quadrature
   treatment, seed61000, 41-pt) — one cheap counterfactual cell.
6. **Cross-cutting — OK with disclosures.** All other tied-to-data numbers in rows
   #127/#128 verify; no gate that should have fired failed to fire. NOT recomputed
   (out of scope): harness-venue magnitudes in rows #120–#124 and csym G-1/G-2 magnitudes
   (#125–#126) — internally consistent with the ledger; the budget correctly excludes them
   per P7-4.

## Overall

The chain is in unusually good shape; T0, regression, counterfactual, and #119 reproduce
exactly and the branch calls are the ones the numbers support. The single item to
reconsider before ratification stands on decision #1 (B-UNOWNED headline residual):
correct r to ≈ +0.056/+0.069, refresh the stale comment, and re-measure s_Edd
production-natively. Decisions #2, #3, #5 and rows #119/#126 can be ratified as recorded.
