# R4 comparand read (verdict-free) — job 6790708 vs S0-B truth vs c0prime_off_iiib

Question of record: morning docket R4 (ledger rows #342 J.2, #345 D3). Retrieval verified by
`batch2_cluster_ops/OPS_RECORD.md` (13/13 md5 byte-identical). No pipeline runs, no cluster; all
numbers below computed locally from the three retrieved `event_likelihoods.csv` files, joined on
`event_idx`, 1588/1588 matched in every pair.

## 1. Provenance table

| flag | comparand (job 6790708) | S0-B truth node (job 6779532) |
|---|---|---|
| git_commit | `40509193...9c70e5` (`run_metadata.json`) | `081b1f28f9d6c36c950954c64f5920f7ea15034d` (`provenance_6779532_4.json`) |
| entry point | production CLI (`main.py --evaluate`) | `fanout1_20260829/hier_s0_driver.py` |
| config/venue | `iiib` (production, no `--config` concept — this IS production) | `iiib` (`build_iiib_venue`, loads same pinned CRB CSV + reduced catalogue) |
| h_value | 0.73 | 0.73 |
| theta_sites | `2.2` | `2.2` (`common_kwargs`) |
| smear_global_selection | n/a (production has no smear flag; `--theta_sites 2.2` alone) | `off` (`_resolve_smear`, forced for sites 2.1/2.2) |
| catalogue_leg_1d_mass_aware | `off` | `off` (driver default, unconditional) |
| theta_b / theta_s | not a CLI flag (production has no theta hook exposed) — resolves 0.0/1.0 | 0.0 / 1.0 |
| theta_phi_divisor | `off` | `off` |
| theta_zwindow | `off` | `off` |
| sky_cone_k | 1.5 | 1.5 |
| catalogue_numerator_survival | not a CLI flag; not in `run_metadata.json`; resolves via `selection_in_completion_numerator`/absolute_marginal path in `evaluate()` | `IIIB_CATALOGUE_NUMERATOR_SURVIVAL = "phi"` (explicit kwarg) |
| catalogue_global_selection | `phi` (CLI arg) | `IIIB_CATALOGUE_GLOBAL_SELECTION = "phi"` |
| selection_in_completion_numerator | `fused` | `IIIB_COMPLETION_CELL = "fused"` |
| mass_filter_geometry / k | `linear` / 1.5 | `IIIB_MASS_FILTER_GEOMETRY="linear"` / `IIIB_MASS_FILTER_K=1.5` |
| **catalogue_numerator_survival_2d** | **not in `cli_args`; `main.py` default = `"mz_sel"`** (production default, [P3-2D] adopted) | **explicitly pinned `"off"`** (`cat_num_surv_2d_kwargs`, driver source: *"every call site below pins the counterfactual explicitly to keep the banked Stage-0/KW-Q1 comparands byte-identical"*) |
| **catalogue_numerator_survival_2d_center** | default = `"eff"` | explicitly pinned `"unset"` |
| mass_filter_sigma | not in `run_metadata.json` `cli_args` at all (no such CLI flag found in `main.py` around the checked args — production resolves whatever `evaluate()`'s own default is) | not passed at the driver's `iiib` call site (function default `"symmetric"`) |

**Difference found (step 4 detail below): `catalogue_numerator_survival_2d` (and its `_center`
companion) — production resolves the *post*-[P3-2D]-adoption default (`mz_sel`/`eff`); the S0-B
driver's `iiib` venue explicitly pins the *pre*-adoption counterfactual (`off`/`unset`) for every
config, disclosed in the driver's own source comment.** No other flag difference was found across
the fields both provenance records expose.

## 2/3. Three-way diff (event_idx join, 1588/1588 matched every pair)

Legend: n>ε = events with |Δ| exceeding threshold; max_rel excludes non-finite (denominator-0) entries.

**(A) comparand vs S0-B truth**
| column | n>1e-9 | n>1e-6 | n>1e-3 | max_abs | max_rel | sign | Σ Δln |
|---|---:|---:|---:|---:|---:|---|---:|
| combined_no_bh | 763 | 390 | 2 | 2.105473e-03 | 4.232e-01 | neg | −2.5800 |
| combined_with_bh | 745 | 282 | 1 | 1.092332e-03 | 6.244e-01 | neg | −5.6140 |
| L_cat_no_bh | 854 | 484 | 39 | 1.300574e-02 | 1.0 | neg | — |
| L_cat_with_bh | 841 | 521 | 26 | 1.762769e-02 | 1.0 | neg | — |
| B_num / B_num_wbh | 43 / 38 | 0 / 0 | 0 / 0 | ~1.4e-7 / 5.6e-8 | ~1.8e-14 | mixed | — |
| D_tilde_phi, alpha_G_phi, w_G, g_frac, den_log_term | 0 | 0 | 0 | 0 | 0 | n/a | — |
| num_log_term_no_bh / with_bh | 896/887 | 562/611 | 144/164 | 0.550/0.979 | 0.036/0.068 | neg | — |

**(B) comparand vs c0prime_off_iiib (theta_sites=all)**
| column | n>1e-9 | n>1e-6 | n>1e-3 | max_abs | max_rel |
|---|---:|---:|---:|---:|---:|
| combined_no_bh, L_cat_no_bh, num_log_term_no_bh, B_num*, D_tilde_phi, alpha_G_phi, w_G, g_frac, den_log_term | 0 | 0 | 0 | **0.0 exactly** | 0.0 |
| combined_with_bh | 738 | 268 | 0 | 2.946795e-04 | 0.162 | Σ Δln = −2.5712 |
| L_cat_with_bh | 829 | 514 | 14 | 4.755438e-03 | 0.934 | — |
| num_log_term_with_bh | 872 | 602 | 145 | 1.771976e-01 | 0.0123 | — |

**(C) S0-B truth vs c0prime_off_iiib — reproduces D3 exactly** (max_rel 0.7338 no_bh / 1.6506
with_bh match the docket's quoted values verbatim): combined_no_bh 763/1588 >1e-9 (390 >1e-6, 2
>1e-3), max_abs 2.105473e-03, Σ Δln +2.5800; combined_with_bh 512/1588 >1e-9 (131 >1e-6, 1 >1e-3),
max_abs 1.089342e-03, Σ Δln +3.0428. `B_num`/`B_num_wbh`/floor columns identical counts to (A)
(43/38 events, ~1e-7/1e-14, mixed sign) — precision-floor noise, not flag-driven, present
identically in every pair.

## 3. g-c0-baseline pattern outcome, pair (A)

**Not GREEN.** max_abs on shared columns ≠ 0 (combined_no_bh 2.1e-3, combined_with_bh 1.09e-3);
the S0-B truth node is **not** byte-identical to the production likelihood at theta_sites 2.2.

**On the no_bh channel, the S0-B/production gap is fully explained by theta_sites**: pair (B)
shows the comparand is byte-identical to c0prime_off_iiib on every no_bh-side column (max_abs
exactly 0.0), so pair (A)'s no_bh diff equals pair (C)'s in magnitude with the opposite sign
(2.105473e-03 both). Set of differing events and the Σ Δln magnitude (2.58 nats) match exactly
between (A) and (C) on this channel — no residual after accounting for theta_sites.

**On the with_bh channel, theta_sites is the dominant but not the sole driver.** (A)'s max_abs
(1.092332e-03) and (C)'s (1.089342e-03) are close but not identical — a residual of order 3e-6 max,
and (B) shows the comparand itself differs from c0prime_off_iiib on with_bh (738/1588 events,
max_abs 2.946795e-04, Σ Δln −2.5712), i.e. **c0prime_off_iiib is not the true production
with-BH baseline either** — it and the comparand differ from each other by the same axis identified
in step 1: `catalogue_numerator_survival_2d`. c0prime_off_iiib long predates the [P3-2D] with-BH
production default; the driver's S0-B `iiib` venue independently pins the same pre-adoption value.
So on with_bh, S0-B and c0prime_off_iiib happen to share the counterfactual `off` setting (which is
why (C)'s with_bh gap ≈ theta_sites-only, same order as no_bh) while the comparand carries the
adopted `mz_sel` default — meaning (A)'s with_bh gap is theta_sites-plus-a-small-2d-twin term, not
theta_sites alone.

## 4. Driver-vs-production flag mismatch (from source, not inferred)

`hier_s0_driver.py` `run_theta_node` (lines ~669-672, 722): for **every** config including `iiib`,
`cat_num_surv_2d_kwargs = dict(catalogue_numerator_survival_2d="off", catalogue_numerator_survival_2d_center="unset")`,
with an inline comment stating this deliberately pins the pre-[P3-2D]-adoption counterfactual "to
keep the banked Stage-0/KW-Q1 comparands byte-identical." `darksiren_emri/main.py` (lines
1425-1426) defaults these to `"mz_sel"`/`"eff"` — the row #223 adopted production value, also
`REGISTERED_RESOLVED_FLAGS`'s implicit basis (that dict does not list `catalogue_numerator_survival_2d`
at all — it predates/omits the 2D-twin axis). No other `EvaluationConfig`-reaching kwarg at the
`iiib` call site (`IIIB_CATALOGUE_NUMERATOR_SURVIVAL="phi"`, `IIIB_CATALOGUE_GLOBAL_SELECTION="phi"`,
`IIIB_COMPLETION_CELL="fused"`, `IIIB_MASS_FILTER_GEOMETRY="linear"`, `IIIB_MASS_FILTER_K=1.5`)
diverges from `REGISTERED_RESOLVED_FLAGS` or from the comparand's resolved CLI defaults.

## 5. Facts for the decider (no recommendation)

- Existence contract: comparand CSV — **PRESENT** (md5-verified, OPS_RECORD.md); S0-B truth CSV —
  **PRESENT** (row #334 md5 match); c0prime_off_iiib CSV — **PRESENT** (used identically in
  `rd-s0b-parity-vs-c0prime/READ_RECORD.md`, re-read here, same 1588 rows).
- Pair (A) comparand-vs-S0-B is nonzero on both channels: this is a genuine, reproducible,
  event-level divergence, not a retrieval or join artifact (join is 1588/1588 exact both pairs).
- The no_bh-channel divergence is attributable 100% to `theta_sites` (2.2 vs the production CLI's
  implicit `all` is NOT what was run here — the comparand *was* run at `theta_sites=2.2`; rather,
  the no_bh gap between comparand/S0-B is byte-for-byte the same as the pre-existing D3 gap between
  S0-B and the `all`-sites c0prime baseline, meaning theta_sites=2.2 production and S0-B agree
  exactly minus what c0prime already differed by — i.e., no new no_bh discrepancy was introduced by
  going CLI vs driver; the driver's no_bh path is faithful).
- The with_bh-channel divergence has two components: the same theta_sites-scale term, plus a
  smaller (~3e-4 max_abs, 738/1588 events) term traced to `catalogue_numerator_survival_2d`/`_center`
  being pinned to the pre-adoption counterfactual in the driver's `iiib` venue versus the adopted
  production default.
- No fix was made; no author ruling issued here.

## CHAIR CORRECTION (2026-09-04 ~02:05) — the reader's step-3 sentence is reversed
Chair re-derived the three-way diff from the raw CSVs (h = 0.73 rows, 1588/1588 matched):
| pair | combined_no_bh n(rel>1e-9) / max_abs / max_rel | L_cat_no_bh | B_num, D̃φ, αφ |
|---|---|---|---|
| A comparand(sites 2.2) vs S0-B truth | 896 / 2.105e-3 / 0.423 | 1083 / 1.30e-2 / 1.00 | identical (B_num 1.8e-14) |
| B comparand(sites 2.2) vs c0prime_off(sites all) | **0 / 0 / 0** | 0 | identical |
| C S0-B truth vs c0prime_off | 896 / 2.105e-3 / 0.423 | 1083 / 1.30e-2 / 1.00 | identical |
Pair B = 0 exactly ⇒ theta_sites has NO effect on the production no-BH likelihood at θ = (0,1)
(identity-forced, as the end-verifier said). Therefore theta_sites explains NOTHING of D3; the S0-B
DRIVER's iiib venue differs from the production CLI in the CATALOGUE LEG (L_cat_no_bh: 1083 events,
max_rel 1.0 = zero-vs-nonzero on some events) while the global-selection/denominator columns are
identical. The reader's step-1 finding names the only flag the driver pins differently:
`catalogue_numerator_survival_2d = off` (production default = mz_sel/eff, the adopted [P3-2D]
value, commit d4765539), with the inline driver comment that this keeps the banked Stage-0 comparands
byte-identical. Discriminating test launched (R4b): production CLI, theta_sites 2.2, mass_aware off,
catalogue_numerator_survival_2d off — if byte-identical to the S0-B truth node, the S0-B measurement was
taken on a PRE-ADOPTION counterfactual of the production 2D setting (a provenance fact for
d-photoz-leverage: the θ-pull read is on "production minus the 2D twin"), not on the production HEAD.
