# r-offset-subset — DESIGN GATE: computability-only review

Reviewer: fresh session, computability lens only. Did **not** open `INFORMATION_FORECAST.md`
(forbidden). Did **not** compute any registered aggregate (AUC/OR/p/Δ_strat) over the registered
1588-event population — every number below is a file/column/log-line/JSON-field existence check,
an md5/byte-id reproduction against the pinned target, or a structural (block-count) check on the
production log; nothing here is a Mann–Whitney U, Fisher exact, or leave-out re-marginalisation
result. Verdict scope: **computability and internal consistency of the registration draft**, not
its scientific merit.

**Overall: GREEN.** Every named input exists, every md5 pin and byte-id anchor reproduces to spec,
every covariate has an executable construction from named columns/functions, and the statistic,
disposition table, blindness structure, and gates are fully specified. Two AMBER
documentation-precision notes below (kill-criterion provenance; a pin-count nit) — neither would
make the eventual read wrong or cause a disposition to bank without author ratification, since the
draft's own binding convention ("nothing frozen until the author rules") already routes every
disposition, including DIFFUSE-IN-COVARIATES, back through `d-offset-subset-register` as a fresh
RULE.

## 1. Named files/columns/log lines/paths — all exist, all pins reproduce

| item | check | result |
|---|---|---|
| production CRB | md5 | `9a1f2a14384a9281c97ca3be312ddaab` — **matches**; 1591 lines = 1590 rows + header — **matches "1590 rows"**. Columns confirmed present: `M`, `luminosity_distance`, `qS`, `phiS`, `delta_qS_delta_qS`, `delta_phiS_delta_phiS`, `delta_phiS_delta_qS`, `SNR`, `in_catalog`, `host_galaxy_index` |
| iiib re-baseline CSV | md5 | `8e6a2c18dc5838dd1d52641589243672` — **matches**; 65109 lines = 65108 data rows = 41×1588 — **matches**. Columns confirmed: `event_idx`, `L_cat_no_bh`, `combined_no_bh`, `combined_with_bh`, `h` |
| joint_r1 replicate CSV | md5 | `745954a0fdee5f10878fb5e622a06144` — **matches** |
| run commit | `GIT_COMMIT_AT_RUN.txt` | `1ec9514dd1808c48b18c0792dce558e5bba0f116` — **matches** |
| catalogue | md5 | `c52c13b5cab61f6b3f04bbe202550969` — **matches** |
| `dark_class.py` | md5 | `841225ac9206ff18bf0145a81cac3a54` — **matches**; both `is_dark_exact(l_cat_no_bh)` and `is_dark_relative(l_cat_no_bh, combined_no_bh, threshold=THRESHOLD)` present with `THRESHOLD = 1e-6`, signatures exactly as C2/C3 use them |
| h=0.73 log, candidate-count lines | line counts | `grep -c "Progess: detections"` = **1588**; `grep -c "no catalog results found"` = **606**; `grep -c "possible hosts found"` = **982** — all three **match §1 exactly** |
| h=0.73 log, P6 line | content | line 8622 reads `P6 host-recovery (h=0.7300): 1D 66/76 hosts recovered/in-cat events seen (86.84211%), 2D 66/76 ...` — **matches "P6 line 8622: 1D 66/76" exactly**, and independently confirms in_catalog=76 (§1 population line) |
| `reduced_galaxy_catalogue.csv` | rows readable | first row `192.721451,41.120152,8.8,0.001733,...` parses; used by C8/cone-radius only |
| jackknife JSON | structure | `results` = list of 4 dicts, one per (venue, channel_label) ∈ {iiib,joint_r1}×{1D,2D}, each with `n_excluded_physics_floor: 0` — **matches G-2(v) "0 events physics-floor-excluded"** |
| `correspondence_1d.py:353` `H_GRID_41` | grid | 41 floats, min 0.60, max 0.86 — matches every `0.60/0.86` rail reference in the draft |
| `tier0_bootstrap_jackknife.py` | functions | `_moments`, `_physics_floor_apply`, `w = np.gradient(h_grid)` (line 37 docstring) all present |
| `cone_loss_reads.py:cone_radius` | signature | `cone_radius(theta, phi_var, theta_var, cov, k)` — matches C5's `cone_radius(qS, δφφ, δθθ, δφθ, k=1.5)` argument order exactly; anchor `"radius": 1.4956979545757095e-03` present verbatim at line 86 |
| `physical_relations.py:447` `dist_to_redshift` | signature | `dist_to_redshift(distance, h=H, Omega_m=..., ...)` — matches C4's call |
| `design_gate_bin_edges.json` `M_edges[2]` | value | `169568.12917853205` — **matches** `--low-m-edge` in §8 and "169 568.13" in §2 exactly |

**Production M range** (independently computed, existence-only, not a registered aggregate):
min 133194.66, max 1627703.88 — confirms §11(e)'s "production M ≥ 1.33e5, edge 1.70e5" and that
only 5 of 1590 rows lie below the C10b edge (< the n≥10 gate, consistent with the draft's own
disclosed C10b near-empty concern).

## 2. Covariates C1–C11 — all fully specified from named columns/functions

Walked each construction against the verified files/functions above:

- **C1** `in_catalog` — direct CRB column. Fully specified.
- **C2** `hosted_exact` — `NOT is_dark_exact(L_cat_no_bh)` on the h=0.73 row of the iiib CSV; function verified to exist with matching signature.
- **C3** `hosted_rel` — `NOT is_dark_relative(L_cat_no_bh, combined_no_bh, 1e-6)`; function verified.
- **C3c** `log10_f_cat` — `L_cat_no_bh / combined_no_bh` at h=0.73, log10, floor −320 for the zero case; fully specified, floor value confirmed below any finite ratio the module could emit (module docstring's own derivation shows the smallest non-zero ratio observed is ~1e-9, ≫ 1e-320).
- **C4** `z_gw` — `dist_to_redshift(luminosity_distance, h=0.73)`; function/line confirmed.
- **C5** `log10_sky_area` — `log10(π·r_cone²)` via the verified `cone_radius` signature; argument mapping (qS→theta, `delta_phiS_delta_phiS`→phi_var, `delta_qS_delta_qS`→theta_var, `delta_phiS_delta_qS`→cov) is unambiguous from the CRB's column names.
- **C6** `mass_window_retention` — `n_1D/n_2D` read off `possible hosts found {len(candidate_hosts)}/{len(candidate_hosts_with_bh_mass)}` (`bayesian_statistics.py:5796`) — confirmed the log literally emits this pair; 46/982 lines carry `n_1D ≠ n_2D` so the covariate is not degenerate. Requires the same log→event_idx block join as C7/G-3a (see below) — computable, not separately gated, but the join is structurally sound (next item).
- **C7** `log10_n_cand_1d` — `log10(1+n_1D)` from the same log line family, `0` when "no catalog results found". Fully specified.
- **C8** `cone_outside` — the r-cone-loss `build_census` OUT flag; function confirmed to exist and take `(crb_path, cat, host_xyz, k)`; restricted to in_catalog (n=76) as stated.
- **C9** alias of C1 — correctly documented as a no-op covariate, not counted toward m.
- **C10** `log10_M` — direct CRB column `M`.
- **C10b** `low_M_timeout_bins12` — `M < 169568.12917853205`; edge value confirmed to exist verbatim in the named JSON; n≥10 conditional gate is well-defined (independently: n=5 in the full 1590-row CRB, so this covariate will very likely resolve NOT-TESTED — consistent with, not contradicting, the draft's own §11(e) disclosure).
- **C11** `log10_snr` — direct CRB column `SNR`, reported-only, correctly excluded from the Holm family.

**Log→event_idx join (G-3a), structural check:** split the log into blocks on `"Progess: detections:"`; 1588 real blocks result (plus one trailing tail after the final detection line, which is not an event block and correctly carries neither marker); every one of the 1588 real blocks carries **exactly one** of `"no catalog results found"` / `"possible hosts found"` (606 + 982 = 1588). The join phase A relies on (block k ↔ scored event_idx k) is therefore well-defined and mechanically executable; G-3a's own decisive count-equality check (606 log hits = 606 `L_cat_no_bh==0` rows) is exactly the right test to catch an ordering mistake, and is itself computable from the pinned files.

## 3. Registered statistics — zero fresh choices

AUC (Mann–Whitney U/(n_S·n_B)), OR with Haldane 0.5 correction + two-sided Fisher exact, Holm
step-down at α=0.05 over the stated m (11 or 10), the AUC(±0.20)/OR([1/3,3]) practical-null band,
the T_mat=0.008 threshold with its explicit sign convention (mean_h<0.73 in every family ⇒
Δ_strat>0 is "toward truth"), the frozen-by-rule stratum definition (binary: enriched level;
continuous: enriched-side decile, n=159), the 1000-draw null with seed 20260904, and the 2-of-3
replicate rule are all stated as concrete formulas/thresholds with no residual free parameter.
`np.random.default_rng(seed)` (not bare `np.random.seed`) is the established convention in this
campaign's own frozen-T0-adjacent scorer (`tier0_bootstrap_jackknife.py:443`), so "seed 20260904"
is not itself an underspecified RNG-API choice.

## 4. Disposition table — three-valued + INSTRUMENT, R14 mandatory, every substantive outcome fresh RULE

SUBSET-IDENTIFIED / DIFFUSE-IN-COVARIATES / INTERMEDIATE / INSTRUMENT-NO-READ. The three
substantive dispositions each end their Action cell "fresh RULE" (§5 table); INSTRUMENT/NO-READ
correctly does **not** carry a RULE action ("repair; no revision consumed") since nothing is
banked in that state — consistent with how sibling nodes in this campaign (`r-cone-loss`,
`r-completion-residual`) treat their own INSTRUMENT-DEFECT rows. The mandatory class-label (R14)
line is present as its own paragraph immediately under the table and is scoped correctly ("evidence
for R14, not the R14 ruling").

## 5. Three-agent blindness + sha256 gate — specified

Phase A (table, never opens the influence JSON top-10 or reference), Phase B (influence, never
opens the covariate table), Phase C (reader, joins after checking the table hash) — §3. G-4 states
the exact mechanism: Phase C refuses to run unless `sha256(covariate_table_blind.csv)` equals the
value Phase A committed to `BUILD_RECORD.md` **before Phase B's first run**, which is the correct
ordering to prevent Phase B from being able to condition on the table. Design-gate reviewers are
separately restricted to a synthetic ≤20-row table (§3, citing the repo's own
`gate-reviewers-must-not-compute-registered-statistic` convention) — consistent with this review's
own operating constraint.

## 6. Build anchors — reproduced from the named JSON, not from anywhere else

| anchor | draft value | JSON value (as loaded) | |Δ| |
|---|---|---|---|
| iiib 2D `full_sample.mean_h` | 0.6658540600 | 0.6658540599535224 | 4.6e-11 (≤1e-9 ✓) |
| iiib 1D `full_sample.mean_h` | 0.6669869414 | 0.6669869414473403 | 4.7e-11 (≤1e-9 ✓) |
| minimal k (iiib 2D/1D, joint_r1 2D/1D) | 82/94/72/46 | `minimal_subset.minimal_k_events_removed` = 82/94/72/46 | **exact** |
| k=1588 endpoint | 0.73 to 1e-12 | `curve_sample[-1]` = {k:1588, mean_h: 0.7299999999997618} | 2.38e-13 (✓) |
| top-10 influence lists | event_idx 576/160/1176 (1D, negative) | `results[iiib,1D].jackknife.top10_events_by_abs_influence[:3]` = [576, 160, 1176], all negative | **exact** |

One path-precision nit (not a defect in the number itself): §1's provenance table lists
`top10_events_by_abs_influence` as if a sibling of `full_sample` and
`minimal_subset.minimal_k_events_removed`; the field actually lives at
`results[i].jackknife.top10_events_by_abs_influence`, one level deeper under `jackknife`, not
under `minimal_subset`. The value reproduces correctly regardless; a builder who greps the JSON
(rather than trusting the described path literally) will not be misled.

Row #344 cone-loss numbers cited in §0 (10 OUT events, φ≈0.4%, leave-out Δmean_h −0.0049, events
889/474 with s_e ≈ +52/−24, SD ratio ≈ 8.5) all reproduce exactly against
`exec/r-cone-loss/READ_RECORD_rev2.md` (Δmean_h,leave-out = −0.004903779..., φ_cone,1D =
0.004335..., event 889 s_e=52.23, event 474 s_e=−24.44, ρ=8.534).

## 7. Kill criterion / max_revisions / blindness-status / md5 pins

- **max_revisions 2** — stated in the header.
- **Blindness-status line** — §10 is thorough: explicitly states the registered statistics have
  not been computed by anyone, names every leak (i)–(iv) with its source row, and states which
  leak bears on a covariate verdict (only (i), C8) and why it cannot alone drive
  SUBSET-IDENTIFIED. This is a genuinely complete disclosure, not a placeholder.
- **md5 pins present** — all five (CRB, iiib CSV, joint_r1 CSV, catalogue, `dark_class.py`) are
  present in §1 and independently verified above. **Minor inconsistency:** G-1 (§6) reads "the four
  md5s and the commit in §1" — §1 actually names *five* md5-pinned artifacts, not four. Cosmetic
  (every pin is still individually specified and will still be checked by a builder reading the
  table), but worth a one-line fix.
- **Kill criterion, verbatim: AMBER.** §5's DIFFUSE-IN-COVARIATES row quotes: *"no single
  registered covariate separates the influence ranking from the bulk at the registered band"*,
  labelled "verbatim from the mandate." A repo-wide grep found this exact phrase **nowhere else**
  in the tree. The only candidate "mandate" — `MORNING_DOCKET_20260904.md` row R1 — reads "route
  the catalogue-hosted-class localization (F1) + the 3–6% subset (F6) into ONE follow-up register
  node (Graph 2 seed)", which is a *routing* instruction, not a stopping rule; no separate Graph-2
  charter document exists yet (r-offset-subset is itself the graph's seed node). Sibling nodes in
  this same campaign (`r-cone-loss`, `r-completion-residual`) treat an unsourced/paraphrased
  "verbatim" kill-criterion claim as a must-fix in their own design gates
  (`r-completion-residual/DESIGN_GATE_design.md` Check 6; `r-cone-loss/DESIGN_GATE_design.md` Check
  6). Here the criterion reads as **orchestrator-derived** (authored by this draft, for this draft)
  rather than truly inherited from an external mandate, and — unlike the bands, the covariate set,
  and the primary-family choice — it is **not** among the six items §9 routes to the author for
  ratification. Severity kept at AMBER, not RED, because the draft's own binding convention already
  routes every disposition (DIFFUSE-IN-COVARIATES included) back through `d-offset-subset-register`
  as a fresh RULE before anything is treated as settled — so a mislabeled provenance tag cannot by
  itself cause an unratified closure. Suggested fix: either cite the actual source of this sentence,
  or relabel it "orchestrator-derived, proposed here" and add it as a seventh §9 ratification item.

## Design-gate self-check cross-reference

Confirms the draft's own §11 self-assessment: (b) "two build scripts do not exist yet" — verified,
`offset_subset_table.py`/`offset_subset_influence.py`/`offset_subset_reads.py` are not present
under `exec/r-offset-subset/`, only the two markdown files. This is disclosed by the draft itself
as the reason it is "un-launchable tonight," not a gap this review is surfacing.
