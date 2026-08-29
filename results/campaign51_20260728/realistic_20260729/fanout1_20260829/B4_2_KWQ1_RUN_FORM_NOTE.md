# B4.2 KW-Q1 run-form note (GAP 11)

**Launched under rows #222/#223 — charter node: NODE archive+minor-notes (wave-2 GAP-CLOSURE),
2026-08-29.**

Purpose: record, per `WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 11, that the KW-Q1 statistic
`s_imp` is invariant to the `theta_sites="all"`/smeared vs `theta_sites="2.2"`/unsmeared run-form
choice, and that the cheaper form is the registered run form. This note does not run KW-Q1, does
not edit `kwq1_score.py` (owned by another agent per the task's standing rules), and does not
change any band or statistic.

## 1. The finding, quoted

`WAVE2_REGISTRATION_CHECK_20260829.md` §3 item 9 ("KW-Q1 (P2) and S0-B form"):

> `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3 names `theta_sites="all"` (smeared) as PRIMARY and
> `"2.2"` as DIAGNOSTIC-NOT-FIX. Chair check of whether F-A contaminates the KW-Q1 statistic:
> `s_imp,i = s_full,i − s_pure,i` with `full = (β_G^φ L_cat + B_num)/D̃^φ` and
> `pure = B_num/D̃^φ` (`kwq1_score.py` header; `bayesian_statistics.py:5770`), so
> `s_imp,i = Δ_h ln[(β_G^φ L_cat,i + B_num,i)/B_num,i]` — **`D̃^φ` and `α_G^φ` cancel identically**
> and `β_G^φ` is θ/smear-inert (`precompute_phi_selection_integrals`, `:4199-4203`). F-A therefore
> does NOT reach KW-Q1's statistic; it reaches only `combined_*` reads (posteriors, the S0-B
> θ-score). The `"all"`/smeared vs `"2.2"`/unsmeared choice for KW-Q1 is a cost (13.7 vs
> 8.4 CPU-h) and CoR-P-fidelity question for any secondary posterior read, not a
> statistic-validity one; the card's labelling may stand. Chair recommendation: run
> `"2.2"`/unsmeared (8.4 CPU-h) and state in the run record that `s_imp` is form-invariant by the
> cancellation above (`L_cat_no_bh` is bit-identical between the forms, §0 F-A).

## 2. Disposition

- **Form invariance:** confirmed by algebra (the chair's derivation above) — `s_imp,i` is built
  from a ratio in which `D̃^φ` and `α_G^φ` cancel identically, and `β_G^φ` is independently
  θ/smear-inert. `L_cat_no_bh` — the remaining ingredient of `s_imp` — is bit-identical between
  `"2.2"`/unsmeared and `"all"`/smeared (`WAVE2_REGISTRATION_CHECK_20260829.md` §0 F-A table:
  `L_cat_no_bh` max_rel 0.0 over the 9 shared events). F-A's non-inertness is entirely in
  `alpha_G_phi`/`D_tilde_phi`/`combined_no_bh` (the `combined_*` posterior reads), which do not
  enter `s_imp`.
- **Run form:** the cheaper form, `theta_sites="2.2"`, `smear_global_selection=False` (8.4 CPU-h),
  is the run form for KW-Q1, superseding `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3's
  `"all"`/smeared PRIMARY designation **for cost purposes only** — the card's PRIMARY/
  DIAGNOSTIC-NOT-FIX *labelling* is not disturbed (per the chair's own text, "the card's labelling
  may stand"); what changes is which form actually gets run, justified by algebra rather than by
  the now-refuted-in-part P1 equivalence.
- **Bands/statistics:** unchanged. No `s_imp` band, threshold, or the KW-Q1 registration's own
  statistics are affected by this run-form choice; this note only fixes which CLI form the run
  uses and records why that choice is cost-neutral to validity.

## 3. Provenance (A11)

| item | value | source |
|---|---|---|
| `s_imp` cancellation algebra | `D̃^φ`, `α_G^φ` cancel; `β_G^φ` θ/smear-inert | `kwq1_score.py` header (not edited by this node); `bayesian_statistics.py:5770,4199-4203` |
| `L_cat_no_bh` bit-identity across forms | max_rel 0.0, 9 shared events, seed 900101, b=+0.02, h=0.73 | `WAVE2_REGISTRATION_CHECK_20260829.md` §0 F-A table, §3 item 9 |
| cost comparison | 13.7 CPU-h (`"all"`/smeared) vs 8.4 CPU-h (`"2.2"`/unsmeared) | `COMPUTE_LEDGER.md` wave-2 cost-refinement row "pre-wave / P2 (KW-Q1...)" |
| PRIMARY/DIAGNOSTIC-NOT-FIX labels (unchanged) | `theta_sites="all"` PRIMARY, `"2.2"` DIAGNOSTIC-NOT-FIX | `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3 |

**Stamp:** launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 11),
2026-08-29. No git operations; no edits to `hier_s0_driver.py` or `kwq1_score.py`.
