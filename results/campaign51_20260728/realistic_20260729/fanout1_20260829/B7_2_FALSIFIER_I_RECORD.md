# B7.2-pre (P4) — falsifier (i) implementation + run record

**Launched under rows #222/#223 — charter node B7.2-pre (P4).** `[FABLE-B7.2-pre 2026-08-29]`

**Date:** 2026-08-29 · **Scope:** implement and run SS6.1 falsifier (i) of
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` (the S_4D-homogeneity / S̄_φ double-weight regression
test; docket §2 B7 condition (a)); append the dated proposal note; no `/physics-change` gate, no
production launch. Builder/runner independence (standing rule 2): unit tests only, no registered
measurement run.

## 1. What was built

`darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py` — 4 test
functions, CPU-only, no GPU/pool/galaxy-catalogue. Reuses
`test_catalogue_numerator_survival_2d.py`'s `_HOSTS`/`_BASE_KW` and
`test_kernel_parity.py`'s `_DETECTIONS`/`_StubDetectionProbability` fixtures per the node
instruction.

**Method** (proposal §1.5/§6.1(i)): a `_ScaledWithBHSurvival` wrapper around
`_StubDetectionProbability` rescales ONLY
`detection_probability_with_bh_mass_interpolated`'s return value by a constant `c`
(`S_4D -> c·S_4D`), delegating every other accessor method unchanged via `__getattr__`. Three
objects are then assembled per the proposal's boxed §1.5 formula
`combined_wbh(c) = (T_cat(mode, c) + T_comp(c)) / D̃(c)`:

- `T_cat` — the REAL with-BH catalogue numerator, via `bs.single_host_likelihood` (production
  kernel, `bayesian_statistics.py:6231-6725`), summed over the three synthetic hosts.
- `T_comp` — the REAL fused completion mass density, via `bs.completion_mass_factor_g_sel`
  (`:2268-2380`), trapezoid-integrated over a 5-node synthetic z-grid, using the SAME wrapped
  accessor as an explicit `s_query` callable.
- `D̃` — a `Σ^4D`-style proxy: the literal per-row point query `S_4D(d_L(z_g;h), M_g(1+z_g))`
  (§1.1) summed over the same three hosts against the same wrapped accessor.

`β_G_φ`/`Σ^φ` are elided as 1.0 (established S_4D-scaling-invariant in ratio, §1.5) since the
falsifiable quantity is `T_cat`'s own degree (0 for "off", 1 for "mz_sel") — the property that is
architecturally unique to this flag; `T_comp`/`D̃` are the SAME code path for both flag values, so
their degree-1 behaviour was measured directly (not assumed) as a sanity check.

## 2. Results (all at HEAD `dd63fe0c`, smoke run 2026-08-29)

| test | result |
|---|---|
| `test_falsifier_i_twin_combined_wbh_invariant_under_s4d_rescaling` | PASS — rel. dev. 2.60e-16 (c=0.4), 1.30e-16 (c=0.15), gate ≤ 1e-10 |
| `test_falsifier_i_coded_combined_wbh_not_invariant_under_s4d_rescaling` | PASS — rel. dev. 1.500 (c=0.4), 5.667 (c=0.15), gate > 1e-3 |
| `test_falsifier_i_detects_double_applied_survival_bookkeeping_defect` (A15) | PASS — synthetic double-survival defect correctly flagged, rel. dev. 0.600, gate > 1e-3 |
| `test_falsifier_i_completion_and_sigma4d_proxies_scale_exactly_with_c` | PASS — `T_comp`/`D̃` linear in `c` to rtol 1e-10/1e-12 |

Full run: `uv run pytest darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py darksiren_emri_test/bayesian_inference/test_catalogue_numerator_survival_2d.py -q`
→ **52 passed** (4 new + 48 existing), 0.96s. `ruff check`/`ruff format --check`: clean. `mypy`:
clean (0 errors after casting `dist_vectorized`'s `floating[Any]` return to `np.float64`).

**Verdict:** SS6.1(i) is a PASS (homogeneity holds; not refuted). Per the falsifier's own
disposition rule this does not return the proposal to the gate — it is the confirming outcome,
not the "structural-omission attribution is wrong" outcome. This closes regression item R3 (§8)
at unit-test scale; promoting it into the default-path regression suite is `/physics-change`
gate scope, not this node's.

## 3. Proposal note appended

`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §13 (dated 2026-08-29, append-only, stamped
`[FABLE-B7.2-pre 2026-08-29]`) now records: (13.1) the falsifier (i) result table above;
(13.2) the STEP-2 smoke item, restated (not executed) — the `h = 0.730` wave-2 task pins the
1.0–1.3× assumed `mz_sel` overhead, expected to scale as `n_cand × 50 × 24`; (13.3) the wave-2 arm
PROD-CF-2D's final registered form (venue iiib; H4 grid {0.660, 0.665, 0.670, 0.730}; gates
R1/R2/R6; `T_mat = 0.008` two-sided; CPU-h 59.7–101.4 nominal, ceiling 105 (arm) / 132 (total)) —
restated from the proposal's own §6.2, confirmed unchanged, not re-derived. No launch was
authorized or attempted by this node.

## 4. Files

- Written: `darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py`
- Appended (§13 only, append-only per standing rule 1):
  `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md`
- This record: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_2_FALSIFIER_I_RECORD.md`

**Stamp:** launched under rows #222/#223 — charter node B7.2-pre (P4).
