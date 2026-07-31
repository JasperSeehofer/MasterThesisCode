# ch07_FLAGS.md — disagreements found while building Chapter 7

Raised by the ch07 agent, 2026-07-31. Per `BOOK_DESIGN.md` §4.1 and the fan-out
brief: where a generator's recomputation disagrees with the build spec or a cited
document, the thread is stopped and recorded. **Nothing below has been reconciled
in either direction.** Both values are carried into `data/ch07_*.json` and both are
visible on the page.

---

## FLAG-1 — σ_dL/dL of EMRI-889 (the running example's dossier row)

| source | value |
|---|---|
| `BOOK_DESIGN.md` §1, Ch 1 card ("EMRI-889 … σ_dL/dL = 8.0×10⁻⁵") and Ch 6 card ("the dossier gains σ_dL/dL = 8.0×10⁻⁵") | **8.0 × 10⁻⁵** |
| `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`, row 889: `sqrt(delta_luminosity_distance_delta_luminosity_distance)/luminosity_distance` | **8.983 × 10⁻⁴** |

Details of the recomputation (ch07 agent, `gen_ch07.py` assertion + manual check):

* Row 889 matches the spec's other identifiers exactly — `M = 724631.5 M☉`
  (spec: 7.25×10⁵), `mu = 10`, `luminosity_distance = 0.0888792 Gpc` (spec: 88.9 Mpc),
  `SNR = 1424.72` (spec: 1425), `host_galaxy_index = 859360` (spec: 859360),
  `in_catalog = True`. So the **event identification is not in doubt**; only the
  quoted distance precision is.
* The value is identical in all six seed61000 CRB copies
  (`seed61000/` and `real_r1..r5/`), i.e. it is not a run-to-run difference.
* Marginal and sky-conditional widths agree to 4 digits
  (8.9833×10⁻⁴ vs 8.9786×10⁻⁴), so the discrepancy is **not** a
  marginal-vs-conditional confusion.
* The independent Gate-B artifact
  `gate_b_20260730/c7_kernel_measure_results.json` stores, for the same host,
  `rel_dL = 8.983284×10⁻⁴` and `sigma_frac_cond = 8.978594×10⁻⁴` — i.e. the
  project's own C7 driver reads the same 8.98×10⁻⁴.
* The median over the 76 in-catalogue hosts is `rel_dL = 5.30×10⁻³`, consistent
  with the ratified `σ_dL/d_L = 0.54 %` golden-set figure in
  `docs/derivations/hostz_pv_photoz_kernel.md` §0 — another sign that the CSV
  scale is the right one.

**Downstream consequence (for the ch01/ch06 agents and the integrator):**
`BOOK_PEDAGOGY.md` Q6.5's answer computes "roughly **6000×** larger" from the
8×10⁻⁵ figure. With 8.98×10⁻⁴ the same comparison gives ≈ 5.5×10² at
σ_z/z = 0.49. Chapter 7 therefore **does not use the 6000× number anywhere**; it
states the comparison from artifacts it can chip (median σ_dL/dL = 0.53 % of this
venue's 76 hosts vs the ratified PV/photo-z dominance measurements of
`hostz_pv_photoz_kernel.md` §0). Ch 1 / Ch 6 should decide the dossier value; Ch 7
prints the CRB value with this flag attached.

---

## FLAG-2 — the C7 rail threshold σ_z/z > 0.256

| source | value |
|---|---|
| `gate_b_20260730/C7_README.md` §1, `CLAIM_2D_BIAS_20260730.md` C7 (as amended 2026-07-30), `gate_b_20260730/ADJUDICATION_20260730.md:148` and `:460` | **0.256** |
| Solving those same documents' **corrected law** `h_eff/h_true = [1+√(1+12ε²)]/2` for `h_eff = 0.86` at `h_true = 0.73` | **0.2644** |

* The threshold is stated identically in three artifacts, so it is not a
  transcription slip inside one file.
* It is **not** computed anywhere in `c7_kernel_measure.py` — the driver only
  records `frac_peak_above_086`; the threshold was evaluated by hand in the README.
* The delivered per-host measurement brackets the same crossing: median peak
  `0.8476` at ε = 0.25 and `0.9390` at ε = 0.35, which interpolates to
  ε ≈ 0.264 for the 0.86 edge — i.e. the **measurement agrees with 0.2644**, not
  with 0.256.
* Neither the superseded 8ε² form (which gives 0.324) nor the small-ε limit 3ε²
  (which gives 0.244) reproduces 0.256 either, so the origin of 0.256 could not be
  reconstructed.

**How Chapter 7 handles it:** the page prints the artifacts' **0.256** as the rail
threshold (that is what the adjudication says, and the chapter may not resolve what
the project has not resolved), draws the reader's live crossing where the quoted law
actually puts it, and carries a visible provenance note stating both numbers and
pointing here. `data/ch07_c7.json → rail_threshold` carries
`artifact_value: 0.256` and `recomputed_from_quoted_law: 0.2644` side by side.

---

## Non-flags — checks that PASSED (recorded so they are not re-run)

These were verified at generation time by `gen_ch07.py` and are reported in its
stdout; they are listed here because a reviewer may otherwise assume they were
taken on trust.

| gate | what was re-measured | result |
|---|---|---|
| G1 | `G2b:288-293` exact posterior-mean shift table at z_g = 0.05 (Ω_m = 0.3) | reproduced to 0.1–0.4 % |
| G2 | `G2b:253-259` amplitude table `C(z̄) = h·s·dln f/dz` | reproduced to < 1 % (only `dln f/dz(0.30)` differs, 3.84 vs the quoted 3.85 — rounding) |
| G3 | the corrected C7 law `[1+√(1+12ε²)]/2` against the delivered per-host median peaks, ε ≤ 0.49 | worst \|Δ\| = 0.0032 (C7_README claims "< 1 % up to ε = 0.5") |
| G4 | the observed in-catalogue ball-numerator tilt, recomputed from `real_r1/diagnostics/event_likelihoods.csv` + `Δln Σ_glob = +0.027597` | max \|Δ\| vs the stored `observed_incat_dln` = **7.2×10⁻¹⁶**; median +0.3082, 93.24 % positive |
| — | `C7_README` §3 quartiles of the indicative σ_z/z over the 76 hosts (0.379 / 0.519 / 0.644) | reproduced exactly from the delivered JSON |
| — | `C7_README` §3 "98.7 % of hosts peak above 0.86" at indicative widths | reproduced exactly (75/76) |

Note on G4: the `diagnostics/` directory is untracked churn and is absent from the
book worktree; `gen_ch07.py` looks in a sibling checkout (relative path only) and
degrades to SKIPPED rather than failing when it is unavailable. The JSON never
depends on it.
