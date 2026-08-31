# END-OF-FAN-OUT VERIFIER REPORT — PART 1 (items 1–19), fan-out 1

**Stamp: registered verifier pass, part 1, 2026-08-30; author check per row #222.**
Registration: `REGISTRATION_END_VERIFIER_PASS_20260829.md` (20 items). Repo
`darksiren-emri`, branch `fix/p32d-classg-venue-repair`. Adoption commit `d4765539`; wave-3
commit of record `60f9996e`. **HEAD at adjudication time is `85dae577`** (the brief named
`b87ad2e6`; `85dae577` is one commit later — C0′ off-gate sbatch, ledger row #253, docket-2
correction note — and changes no verdict below). Adjudicator: one top-tier agent
(`xhigh`), foreground only, no `ssh`, no `git commit`/`add`, no edits to code, registrations
or records; this file is the only write outside `verifier_pass/`.

**Item 20 (the wave-3 blind HEAD readout) is DEFERRED — cluster SSH has been down since
≈21:15 on 2026-08-29; wave 3 is built (`60f9996e` + `85dae577`) but NOT submitted.** Part 2
appends to this report once it lands (§5).

**Counts (items 1–19): confirmed 18 · refuted 0 · undetermined 1 (item 19) · deferred 1
(item 20). Author items returned: 17 ([RULE] 12 · [DO] 4 · [STANDING] 1) + 10 orchestrator
path decisions of record open to veto.**

### Disclosures on how this part was produced (read before the table)

1. **Items 12–19 reached the adjudicator without verifier verdicts.** The payload handed to
   this task contained complete structured verdicts for items 1–11 and a fragment of item 12
   (cut off inside `author_rule_items`); the dedup/conflict memo
   (`verifier_pass/DEDUP_CONFLICTS.md`) likewise covers items 1–11 only. The eight verifier
   scripts for items 12–19 (`verifier_pass/item1{2..9}_rederive*.py`) were on disk, so the
   adjudicator **executed each of them itself, read the source records and code they
   cite, and ruled from that** (the brief authorises re-derivation for any item without a
   usable verdict). Consequence: for items 12–19 one agent performed both the verifier and
   the adjudicator role — the registered two-layer independence (sonnet verifier → top-tier
   adjudicator) is reduced to one layer. The author may order a supplementary sonnet panel
   on items 12–19 if the two-layer form is wanted; the numbers below would not change, only
   their attestation depth.
2. Registration §3 says the adjudicator "does not re-run any computation itself"; the brief
   for this task overrode that for refuted/undetermined/conflict items and, by the gap in
   (1), for items 12–19. Every re-run is foreground, against local CSV/JSON/log/code, and
   listed per row.
3. No verifier returned `refuted`. Per registration §5 the expected first-pass catch rate
   is ≈80 % on citation/completeness-class items; that is what happened — 13 of 19 items
   carry at least one must-fix-class note, none reverses a headline number or verdict.
4. Caps are carried verbatim from the records: REPORTED-ONLY ([HIER] PA-HIER-28 item 9;
   KW-Q1; [CMEM] structural), `supported` (B7 calibration status), PROVISIONAL (C4/B7.3
   attribution until falsifier (ii)), [LOCAL] (B4.1 forecast inputs).

---

## 1. Verdict table

Columns: item · node · verdict of record as claimed (with its cap) · verifier verdict ·
decisive number **claimed → re-derived** (source re-opened) · cap carried · adjudication note.
"(adj.)" marks rows where the adjudicator ran the re-derivation itself (disclosure 1).

| # | node | verdict of record (cap) | verifier verdict | decisive number claimed → re-derived | cap | adjudication note |
|---|---|---|---|---|---|---|
| 1 | B1.1 wave-1 record (`B1_1_HIER_RECORD.md` + build note) | B0-A/A′ UNDETERMINED at 1 seed; θ-engaged smeared cell 18.6× the §7.1 anchor; site 2.3 inert for the no-BH channel (REFUTED-IN-PART by F-A for `combined_no_bh`); GATE PARITY residual 5.718e-4; 4 must-fix citations appended (REPORTED-ONLY) | **confirmed** | 1190.93 s vs 64.73 s, 18.6× of 63.97 → **18.617×** from `s0a_seed900101_full.log` (timer wraps only `bs.evaluate()`, `correspondence_1d.py:2930/2967`); PARITY `combined_no_bh` max_rel 5.718e-4 → **5.718020e-4** (own pandas merge of the two raw CSVs); ternary regex-matched at HEAD `:5215` (was `:5187-5191` at ff230621, +28 drift); F-A mechanism re-derived from `:5775/:5797` (D̃_φ from `global_denom_with_bh`); T-ID 20/20 re-run | REPORTED-ONLY carried | Verifier-found: the appended must-fix item 3 cites `hier_s0_driver.py:242-245` for the `np.where(vals>0, np.log(vals), nan)` guard — **that citation is itself wrong** (those lines are a docstring at ff230621; the guard is at `:425`, now `:444`). Adjudicator re-checked: confirmed. A citation error inside the citation-correction note; substance (natural log) correct. Append a one-line fix. |
| 2 | B1.1 Stage-0 (`B1_1_HIER_STAGE0_RECORD.md`, prereg `:2526-2683`) | B0-A′ → INSTRUMENT-DEFECT → STOP (REPORTED-ONLY); dark class score exactly 0 | **confirmed** | Z_b −3.676, Z_s −7.079 (N=461, 4 seeds) → **−3.676431 / −7.078607** from the 20 raw `event_likelihoods.csv`; per-seed Z reproduced to 6 s.f.; dark class n=5 → 0.0 exact; ENG 0.98858 | REPORTED-ONLY carried | Registration text "no forensic on disk" is stale — `B1_1_S0A_DEFECT_FORENSIC_20260829.md` landed in `b87ad2e6`; it does not alter the STOP. Open: R2′ (CoR-M) [RULE], S0-B word, §4.5 fix routing. |
| 3 | B2.1 [CMEM] A1 (`B2_1_CMEM_A1_RECORD.md`, `cmem_a1.py` sha1 `75751f3c…`) | R2c NOT-DISTINGUISHED, parked; C-STRUCTURAL-ONLY stands (REPORTED-ONLY/structural, single-h) | **confirmed** | T=−0.12311421153794763, p=0.0358; T_w=−0.10828010490112266, p=0.0522; census 380/2336=0.16267 → **bit-identical**, fresh census build from the source CRB/diagnostics CSVs (seeds 900101–110 present, 900111/112 absent both arms, on disk); catalogue md5 `c52c13b5…` re-hashed | REPORTED-ONLY / structural carried | Cap citation "row #216 item 4" → row #219 already appended (append-only, verified). No pooled statistic exists anywhere (confirmed by grep) — docket §6 item 7 premise holds. |
| 4 | B3 (`B3_1_POP_RECORD.md`, `B3_2_POP_FLAG_RECORD.md`, `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md`) | closure PREMISE-REFUTED (provenance, zero compute); STOP correctly declined by the builder | **confirmed** | `git show 03cfe80:…/dark_siren_injection.py:328` = `(1−f)·_redshift_population_weight` = current tree `:328`, byte-identical to the estimator's `1/(1+z)·dVc/dz` prior (`bayesian_statistics.py:1170-1216`); HEAD dark-class score −0.4668±0.0162 / −0.3938±0.0207 vs −0.635/−0.565 → **7.1637σ / 5.9522σ**; five-bin counts 605/491, underflow 1/2 → exact; CRB md5 `9a1f2a14…`, 1514 dark / 76 in-cat re-read | none (no band) | Append-only discipline verified across dd63fe0c→ff230621 on all three touched docs. G7 row 16 re-grade → [RULE] A5. |
| 5 | B4.1 [IMP] (`B4_1_IMP_RECORD.md`, decomposition, `CLAIM_IMPOSTOR_DRAG_20260829.md`) | NOT EXONERATED; remainder not diffuse; NECESSARY cause of the 1D rail (ASSUMPTION-JOIN); mechanism UNDETERMINED ([LOCAL] inputs) | **confirmed** | Δ_FT +0.12274±0.00774 (80.8 % of +0.15181) → **+0.122745±0.00774**, 12/12 positive (own assembly from raw `event_likelihoods.csv`, pipeline `combine_log_likelihood`); production iiib pure-dark-only mean_h 0.7134 / MAP 0.70 / σ 0.0277 → bit-identical re-run against `headreadout_20260827`; O2 4e-17 → 4.16e-17; 76/1588 join exact; all 17 exoneration citations land at the cited lines | [LOCAL] carried | Row #167 (D̃_φ sub-convention) still open — no later ledger row (grep). B4.3 per-candidate hook needs a physics-scope word → [RULE] A13. |
| 6 | B4.2 KW-Q1 (`B4_2_KWQ1_READOUT_RECORD.md`, `b4_2_readout.json`) | KERNEL-WIDTH-INERT, REPORTED-ONLY (instrument-defect disclosure carried); A14 falsifier not withdrawn | **confirmed** | R=+0.08481225026529439 → **bit-identical** (third implementation from the 12 raw node CSVs); S(1/√2)/S(1)/S(√2) −1.0456670/−1.0205308/−0.9591134 exact; per-seed max 0.156 < 0.2; GATE I 7.61e-8; ENG 486/486; T-ID parity max|Δ| 0.0 over 348 rows; q1 share 92.25 % | REPORTED-ONLY carried | Forensic E21 quantifies a divisor-gap contamination of R ≈ +0.02–0.04 not folded into the record (docket 2 §5 item 2); INERT survives either sign (|R| < 0.13 worst case). The run-form note's "form-invariant" is exact for D̃_φ/α_G^φ, first-order for the divisor gap — record should carry that narrowing (must-fix, append). |
| 7 | B5.1 [WIN] (`B5_1_WIN_RECORD.md`, `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`, `0b308828`) | flag implemented, default byte-identical; zero-compute count refuted-in-direction (log k3 reduces candidates, retention 0.957→0.789) | **confirmed** | pass fraction 0.95768 / 0.69509 → **0.9576806 / 0.6950869** via the *production* `get_possible_hosts_from_ball_tree` on the real 24-arm fleet; retention 0.9567→0.7890 → exact; 100 000-pair byte-identity → 0 mismatches; 24 tests pass | none | Verifier-found: runbook 37 §5 and the gate-doc header label a paraphrase of row #223 as "(author, verbatim)"; the ledger's verbatim text differs in wording (substance identical). Adjudicator re-checked: the quoted sentence does not occur in row #223. Attribution-precision slip → append correction. |
| 8 | B5.2 C3 (`B5_2_WIN_K3_READOUT_RECORD.md`, `b5_2_readout.json`, `B5_2_PULL_READ_20260829.md`) | INTERMEDIATE, REPORTED, adoption NOT granted; R1 retention transfer FALSIFIED (66/76 both arms); pull-read/L9 reconciled | **confirmed** | Δmean_h,pred +0.0035225270694619775 → **+0.0035224477** (2.3e-5 rel, INTERMEDIATE either way) from the two raw CSVs; R6 2.667e-14 → 2.41e-14; R2 982/951 exact; R5 −63.7051 exact; 66/76 arm-side from `wave2_c3_task3_6738999.err:9917`, baseline-side independently corroborated by `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13 (job 6738998); 621 collapse events 100 % dark-class; `b5_pull_read.py` re-executed end-to-end, output bit-identical | none | The dedup memo's one flagged "conflict" (item 7's 0.789 vs item 8's 66/76) is **resolved: two different statistics** — 0.789 is the mirror fleet's true-host retention under the mirror's linear-Gaussian mass law (`b5_window_count.py` over `bc_9001XX` arms); 66/76 is production iiib in-catalogue recovery. Item 8's "falsifies" refers to the *transfer* of the mirror number to production, not to item 7's arithmetic. A22 "tree clean" vs `tree_dirty_file_count=296` remains an assertion (C0 §13), see §3. |
| 9 | B6.1 [ALIGN] (`B6_1_ALIGN_RECORD.md`, `1f003da6`) | [PHYSICS] commit; s scales the raw σ_z before the PV fold; bit-identical while `SIGMA_V_PEC_KM_S=0` | **confirmed** | targeted 27/27; full suite 1896/15/27 (record 1851 — +45 tests from sibling nodes, 0 failed both); discriminator at σ_pv=200 km/s, s=1.4142 matches the pre-fold closed form at rtol 1e-9 and does NOT match the post-fold form (0.0424387 vs 0.0424323 — non-vacuous); hooked call matches the raw-z reading, not the z̃ literal; `git log d04d9dc9..dd63fe0c` = exactly d40fe5c8, 1f003da6, 0b308828, 901653a1 | none | The raw-z-vs-z̃ judgment call is the builder's, disclosed, not yet ruled → [RULE] A12 (verifier finds it consistent with prereg §1.2's unshifted `z_centre`). |
| 10 | B7.1 [2D-TWIN] proposal (`PROPOSAL_2D_TWIN_ADOPTION_20260829.md`) | `eff` centering decided in-proposal, numerically inert (σ_cond p50 8.8e-8); cost 74.7–101.4 CPU-h; S-homogeneity deferred to falsifier (i) | **confirmed** | σ_cond p50 8.8e-8 → **8.796e-8** (rebuilt from `proda0_work` CRB with the production `Detection`/`cov_4d` formula; cross-venue, ratio 1.000); cost arithmetic from `LAUNCHING_JOBS.md:47` exact; falsifier (i) 2.597e-16 / 1.298e-16 / 1.500 / 5.667 / 0.600 reproduced | `supported` cap carried | Verifier-found: no builder-report/verifier-report artifact exists on disk for the "panel clean after 0 rounds" claim (only restated record→docket→row #231). Proposal §2.2's "<~1e-14" understates its own band top (8.6e-14) — immaterial. |
| 11 | B7.2 C4 (`B7_2_TWIN_CF_READOUT_RECORD.md`, `b7_2_readout.json`, falsifier (i) record) | IMMATERIAL-PREDICTED +0.0025057; R1/R2/R6 PASS; PROVISIONAL (provenance extras; falsifier (ii) unrun) | **confirmed** | Δmean_h,pred → **0.002505684643832008, diff 0.0** from the two raw CSVs (Δℓ′ 7.429354968961904; Δℓ″ −30.31136388614181); R1 0/6352 (2424 empty-equal); R2 982/982; R6 0.0 all 4 nodes; falsifier (i) 4/4 with matching rel_dev | PROVISIONAL carried | C4 `run_metadata_*.json`, `logs/`, `GIT_COMMIT_AT_RUN.txt`, with-BH posteriors h=0.67/0.73 confirmed absent locally (SSH) — gates rest on the diagnostics CSV, which is complete. |
| 12 | B7.3 adoption (`PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md`, implementation record, verifier report, `d4765539`) | ADOPTED as production default (`mz_sel`/`eff`), structural consistency, no bias claim; pending wave-3 readout + `off` baseline; attribution PROVISIONAL; `supported` | **confirmed (adj.)** | `item12_rederive.py` exit 0: commit touches exactly the 4 named production files (+ tests, records, driver, 5 scripts; 16 files); all 5 `bayesian_statistics.py` hunks at 3268–3726, **outside** the kernel range 6231–7723; **12/12 pin tests PASS** incl. `test_off_matches_the_pre_flag_golden_across_modes[×3]` and `test_kernel_default_pair_bit_identical_to_explicit_mz_sel_eff`; full suite **1896 passed / 15 skipped / 27 deselected** (79 s); five archived scripts pin `"off"`/`"unset"` at the cited lines; ruff/py_compile clean | PROVISIONAL + `supported` carried | Housekeeping still open: `(row #<adoption>)` placeholder at `bayesian_statistics.py:3274` (row #253 now exists); INFO line at `:3728` reads "adopted under row #223" while §13.1/verifier report quote "ACTIVE (row #249)" — append a reconciling note. Ratification → [RULE] A4. |
| 13 | B8.1 [CAL] floor (`B8_1_CAL_FLOOR_RECORD.md`, `b8_information_floor.{py,json}`) | σ_h,floor(1D, σ_z=0.035)=0.001747; 2D at σ_M=1.99 identical to 4 s.f.; Route A unstable (0.000371, n_eff≈5) = negative result | **confirmed (adj.)** | independent explicit 2×2 Fisher in (h,z) with numerical derivatives of the production `dist_vectorized` + hand Schur complement: **0.001747 / 0.001747 / 0.001295 (σ_M=0.02) / 0.000560 (spec-z)** — all four exact; factorisation d_L=D(z)/h confirmed to 9.8e-11; the disclosed "both terms × h²" slip re-introduced deliberately gives floor ×0.799 (info overstated 1.57×, inside the script's "up to 1.9×"); Route-A instability reproduced by a different estimator (event idx 889, z=0.0213: local quadratic curvature negative, wide-window positive) | [INFO-STARVATION] not resurrected (register §13) — carried | Width/floor 10.57×, |bias|/floor 38.24× reproduced. No must-fix. |
| 14 | B8.2 [CAL] harness design (`B8_2_HARNESS_DESIGN_20260829.md`) | honest cost correction 130–475 CPU-h local (20–80× the docket's ≈6 CPU-h) — design note, no band | **confirmed (adj.)** | docket anchor 24×65 s×14 = 6.07 CPU-h ✓; production-N linear extrapolation 969.7 s / 3.77 CPU-h ✓; **correction factor 20.6–77.7× ✓**; but the note's own rows sum to **125–471** (cell S+T) or **160–513** (all rows) — "130–475" matches neither; "13–46 h wall" ≠ CPU-h/14 (8.9–33.7 h) | design note only; bands present as **disclaimed placeholders** (§4 "placeholders for the registration"; line 7) — no A15 power claim smuggled | Must-fix (append): reconcile the mandatory total and the wall-time line with the table; state the 1.0 CPU-h/universe floor as a judgment value. Decisive claim (order-10² CPU-h, 20–80×) stands. |
| 15 | C0 baseline gate (`REGISTRATION_C0_BASELINE_GATE_20260829.md`, job 6738998) | PASS bit-identical; costing anchor 9–13× overestimate; anchor from the 3355-event set | **confirmed (adj.)** | **max_abs = 0.0 on all 14 columns, 1588 events, 22 232 value pairs, 0 NaN mismatches**; both posterior JSONs md5-identical (`563ef45b…`, `2b4fb3e0…`); c0 carries exactly the 3 extra OAT columns; 1.7244 CPU-h vs 15–23 → **8.70–13.34×**; `LAUNCHING_JOBS.md:47` anchor reads "@ 3355 events" ≠ 1588; bonus: the §11.2 OAT identity check (docket 2 listed it as NOT RUN) **executed here: max_abs 3.1e-15 both channels, PASS** | none | Elapsed 00:06:28 is taken from the record (`sacct` not re-pullable) — that primitive is the item-19 UNDETERMINED. |
| 16 | B1.2 PA-HIER-31 (+ REVISION NOTES 1–2; prereg `:1951-2500`) | S0-B registration; F-A divergence; open items routed as fresh [RULE]s | **confirmed (adj.)** | seed 900101, 9 shared events, b=+0.02, h=0.73: `L_cat_no_bh` max_rel **0.0**; `combined_no_bh` **7.447115e-3**; `alpha_G_phi` 5.8688310e7→5.1635200e7 (**−12.018 %**), `D_tilde_phi` 9.470921e8→9.40039e8 (**−0.745 %**), `w_G` 0.06196684→0.05492879 — all exact, all per-event-constant | registration, no band | **Registration text is stale on one point**: "original register's author item 3 … deferred" — PA-HIER-28 (author verbatim "exactly as recommended by you") resolved items 3=GATE, 4=GATE, 5=FALLBACK, 9=AFFORDABLE; item 7 executed (PA-HIER-29), item 2 approved (row #216 (5)). Adjudicator's completed count of the [HIER] return set: **R1 (CoR-P), R2′ (CoR-M), R4′ + the CoR-P P1′ (two unexecuted 0.33 CPU-h measurements), item (f) residual disposition (now diagnosed: 401→4001 grid), and the forensic's two amendment items (secant form; z-binned read)**. Completed count of the pre-existing nine "(ii) OPEN-FOR-AUTHOR" items: **all nine are closed** — items 1 (venue b0i RATIFIED), 2 (θ hook APPROVED), 6 (θ prior option B RATIFIED), 8 (h support `H_GRID_41` RATIFIED) in PA-HIER-27; items 3, 4, 5, 9 in PA-HIER-28; item 7 executed in PA-HIER-29. The [HIER] return set is therefore exactly the post-hoc items just listed (A2), none of the original nine. |
| 17 | Path choices + tree state (docket 1 §5 items 9–10; row #238 §4; row #239) | AGREE 7/10, DEVIATE on B1/B3/batch, each with a number; dirty-tree byte-identity argued not stamped | **confirmed (adj.)** | `WAVE2_REGISTRATION_CHECK_20260829.md` §4 re-read: rows 1, 3, 9 deviate, each with a citable number (F-A 7.45e-3 / α_G^φ −12 %; §F provenance md5 `9a1f2a14…` 1514/76; §3 item 7 → 13 tasks 179–357); rows 2, 4–8, 10 AGREE. F-A itself re-derived (see 16). Dirty tree: B6.1/B5.1 committed 19:06:02/19:06:22 on 2026-08-29, S0-A wave-1 started 17:58 → ran on uncommitted edits; byte-identity at the measured nodes is now **stamped, not merely argued**: item 7 (100 000 pairs, production function), item 9 (θ=(0,1) pins; s=1 is an algebraic no-op at σ_pv=0 for every node), item 15 (C0 bit-identical at production scale against the pre-edit bank) | none | Provenance integrity holds; the docket's "argued, not stamped" wording can be retired by an appended note citing items 7/9/15. |
| 18 | Governance incidents (docket 1 §7, docket 2 §8; runner logs; `COMMIT_PLAN_3.md` §4–5) | 12+15 incidents disclosed; runner-1→2→3 chain; SSH interrupted C4 retrieval; ≈93.5 MB intermediates excluded by path-filtered add | **confirmed (adj.) — with one new finding** | From the three raw runner logs: runner-1 `ValueError: No objects to concatenate` on an empty `all_nodes["b_plus"]` list ✓ (rc=1 ×2); runner-2 `AssertionError: daemonic processes are not allowed to have children` at `bayesian_statistics.py:4562` in `evaluate` ✓ (4 worker errors); runner-3 `--jobs 1` (JSON `jobs: 1`; label string "jobs2" cosmetic) rc=0 ×4 ✓. C4: diagnostics CSV present (6 353 lines); `posteriors_with_bh_mass/h_0_67,h_0_73` absent; `run_metadata_*.json`, `logs/`, `GIT_COMMIT_AT_RUN.txt` absent ✓ (genuinely not retrieved). Commit hygiene: **largest blob ever committed under `hier_s0_*` = 8 438 bytes**; no `simulations/` path tracked ✓ — no sweep-in | none | **NEW (verifier-found, not disclosed):** `.gitignore:1 (*.log)` and `:16 (simulations/)` block exactly the slice `COMMIT_PLAN_3.md` §4 said would be committed ("only `*.log` files and per-node `diagnostics/event_likelihoods.csv`"). What is tracked under `hier_s0_registered_run/` is 9 files (nohup.out, stage.txt, score md/JSON); **none of the 41 raw files (20 S0-A + 13 KW-Q1/parity per-node CSVs + 8 runner/full logs; 6.4 MB total) is in version control**, and `kwq1_registered_run/`, `kwq1_parity_run/`, `b1_1_forensic_work/` have 0 tracked files. Docket 2 §8 item 4's sentence "the path-filtered add … was applied (the wave-2/3 commits carry logs, small CSVs and JSON only)" is **inaccurate in the inclusion direction**. These local runs have no cluster copy. → [DO] A15. |
| 19 | Compute ledger + F4 (`COMPUTE_LEDGER.md`; rows #246–#249, #252) | wave-2 cluster 13.5 CPU-h vs 179–357 (13–26× favourable); KW-Q1 6.152 vs 8.4; F4 deadline clear | **UNDETERMINED (adj.)** on its registered criterion; arithmetic confirmed | From the quoted Elapsed strings: C0 1.7244 + C3 4.9733 (290+279+276+274 s = 1119 s, matching the docket-2 correction) + C4 6.8000 = **13.4978 CPU-h**; 179/357 ÷ 13.4978 = **13.26–26.45×** ✓; per-arm 8.7–27.6× ✓; KW-Q1 **6.151663 CPU-h** from `b4_2_readout.json:cost_measured` (local source) ✓; P0 11.51 / S0-C 10.42 from `s0a/s0c_full_output.json` ✓; fan-out total 52.98 ✓; wave-3 159.77–290.10 ✓; workspace expiry 2026-09-23 = **24 days** clear ✓ | none | The registered refutation criterion is "a measured-cost figure does not match its cited `sacct`/log source". **The C0/C3/C4 `Elapsed` primitives exist locally only as quoted strings** (`b5_2_readout.json:cost_F4.measured_tasks`, `run_metadata_21.json` carries the job id but no elapsed, the c3 `.out/.err` logs carry no timing line); `sacct` cannot be re-pulled (SSH). The ledger's arithmetic and every locally-sourced figure reproduce; the three cluster primitives are unverifiable at zero compute. Note the 179–357 band includes C1 (never launched); against the launched-arms-only band (119–265) the miss is 8.8–19.6× — still favourable. Close in part 2 with `sacct -j 6738998,6738999,6739000,6739001 -X --format=JobID,Elapsed,AllocCPUS`. |
| 20 | Wave-3 blind HEAD readout | not yet run | **DEFERRED** (SSH down) | — | — | see §5 |

---

## 2. REFUTED and UNDETERMINED items, in plain language

**No item is refuted.** Every decisive number in items 1–18 was re-computed from the raw
artefact the record cites (CSV, JSON, log, git object, or production code) and reproduced to
the record's stated precision — in most cases bit-for-bit.

**Item 19 — UNDETERMINED (compute ledger, cluster arms).**
*What was claimed:* the three wave-2 cluster arms cost 1.72 + 4.97 + 6.80 = 13.5 CPU-h, 13–26×
below the registered estimate, and the F4 deadline gate is clear.
*What the re-derivation shows:* all of the arithmetic is right, every locally measured cost
(KW-Q1, P0, S0-C — read from their own JSON outputs) is right, the wave-3 estimate is right,
and the workspace has 24 days left. But the three cluster numbers rest on SLURM `Elapsed`
strings that were read off `sacct` on 2026-08-29 and copied into the records; no raw `sacct`
dump was saved locally and the cluster is unreachable, so nobody in this pass could open the
source the ledger cites. The item's own registered test ("does the figure match its `sacct`
source?") therefore cannot be run.
*What it changes:* nothing about any verdict (no verdict depends on a cluster CPU-h figure),
and nothing about the F4 deadline. It is a provenance gap in the F4 instrument, closable in
minutes once SSH returns (part 2).

**Sub-claims that were refuted inside otherwise confirmed items** (each is a citation,
arithmetic-consistency or completeness correction; none moves a headline):
- Item 1: the must-fix note's own citation for the ln-transform guard (`hier_s0_driver.py:242-245`) is wrong; the guard was at `:425` when the note was written.
- Item 7: a paraphrase of row #223 is labelled "verbatim" in two documents.
- Item 14: the design note's "130–475 CPU-h" and "13–46 h wall" lines do not follow from its own cost table (125–471 for the mandatory cells; wall ≠ CPU-h/14).
- Item 16: the verifier registration's claim that PA-HIER author item 3 is "explicitly deferred" is stale — the author resolved it (GATE) in PA-HIER-28.
- Item 18: docket 2 §8 item 4's statement that the wave-2/3 commits "carry logs, small CSVs" is wrong — `.gitignore` excluded both classes, so the registered local runs' raw per-node CSVs and logs are not in git.
- Item 10: proposal §2.2's "< ~1e-14" understates its own derived band top (8.6e-14); immaterial.

---

## 3. Governance breaches — disclosed by the records vs found by the verifiers

None of the entries below reverses a verdict. "Severity" follows registration §5's scale
(citation/completeness < scope < provenance < production-flip).

### 3.1 Disclosed by the records themselves (docket 1 §7, docket 2 §8, node records)

| # | incident | where disclosed | status after this pass |
|---|---|---|---|
| D1 | B1.1 wave-1: registered scope not completed; two orphaned background runs (≈20–30 CPU-min unbanked) | docket 1 §7.1 | completed by P0 (runner-3); costs now in the ledger except the crashed attempts (D11) |
| D2 | B1.1: four must-fix citation errors, appended not edited | docket 1 §7.2, B1.1 record | verified appended; one new citation error inside the correction (item 1) |
| D3 | S0-A wave-1 measured on a dirty tree (B6.1/B5.1 uncommitted edits) | docket 1 §7.3 | byte-identity now stamped by items 7/9/15 (item 17) |
| D4 | Builder-run smoke tests / no refuter report reached the chair for B5.1-impl, B6.1, B8.1; B7.3 "verified" ledger row builder-run | docket 1 §7.4–7.6, §7.10; docket 2 §8.9 | this pass supplies the missing independent re-execution for each (items 7, 9, 13, 12) |
| D5 | B3.2 dispatch text instructed implementation against a STOP; builder declined correctly | docket 2 §8.12, row #234 | closed; correct application of the approval-scope rule |
| D6 | B2.1 inherited cap citation (row #216 item 4 → #219) | docket 1 §7.8 | appended correction verified (item 3) |
| D7 | Runner-1 → runner-2 → runner-3 crash chain; `--jobs>1` dead in the driver | docket 2 §8.1 | diagnoses confirmed from raw logs (item 18) |
| D8 | SSH outage: C4 provenance extras, C0 `sacct` re-pull, archive script, wave-3 submission all blocked | docket 2 §8.3 | confirmed; drives item 19's UNDETERMINED and item 20's deferral |
| D9 | GitHub 100 MB rejection → two amended commits → `.gitignore:101-102`; no ledger row names the event | docket 2 §8.4 | still no row; recommend the one-line note ([DO] A16) |
| D10 | B7.3: log-line citation changed before commit; `(row #<adoption>)` placeholder at `:3274`; adoption ledger row missing | docket 2 §8.5–8.6 | row #253 now filed (`85dae577`); placeholder and record/code text mismatch still open ([DO] A16) |
| D11 | ≈8.6 CPU-h of crashed runner attempts unbanked | docket 2 §4 | still unbanked; add a ledger line ([DO] A16) |
| D12 | `tree_dirty_file_count=296` vs A22 "tree clean" stamp | row #247, docket 2 §8.8 | see F5 below — the "resolution" is an assertion |
| D13 | Forensic E7 overclaim ("own GL-50 quadrature" reuses production helpers) | docket 2 §8.11 | not corrected (append-only); does not touch 9.2e-13 |
| D14 | Verifier-registration staleness: item 16 "author item 3 deferred"; item 2 "no forensic on disk" | docket 2 §8.13 | both confirmed stale (items 2, 16) |
| D15 | B6.1 judgment call (raw z vs z̃) disclosed in three places | docket 1 §7.5, B6.1 record §3 | open → [RULE] A12 |
| D16 | KW-Q1 first scorer invocation found 0 rows (path error), re-invoked | row #249, docket 2 §8.2 | closed; excluded from cost |
| D17 | Top-tier cap ≤3 per wave respected; no rule-6 breach; no node ran git | docket 1 §7.11–12, docket 2 §8.14 | consistent with what this pass saw (gated commits 1f003da6, 0b308828, d4765539 only) |

### 3.2 Found by the verifiers (not disclosed in the records)

| # | finding | item | severity | recommended action |
|---|---|---|---|---|
| F1 | Citation error inside B1.1's citation-correction note (`hier_s0_driver.py:242-245` → `:425`) | 1 | citation | append one line |
| F2 | "(author, verbatim)" label on a paraphrase of row #223 in runbook 37 §5 and the mass-window gate-doc header | 7 | attribution precision (CLAUDE.md convention) | append a correction naming the true verbatim text |
| F3 | B7.1 "panel clean after 0 rounds, two independent reports" has no report artefact anywhere under `fanout1_20260829/` (`verify_b51/` is empty); the claim is only ever restated | 10 | completeness / rule-2 evidence | either file the two reports or re-label the panel state "unfiled" |
| F4 | The `.gitignore` rules `*.log` and `simulations/` defeated COMMIT_PLAN_3's intended slice: **41 raw files (6.4 MB) of the registered local measurements (S0-A, KW-Q1, parity) exist only on the local disk** — no git copy, no cluster copy, not in the Option-A archive; docket 2 §8.4 states the opposite | 18 | provenance / archival | [DO] A15 |
| F5 | The A22 "clean tree" stamp in two registrations is contradicted by `tree_dirty_file_count=296` on every C0/C3 provenance JSON; C0 §13's "untracked-file count" explanation is unverified by any agent (SSH) and docket 2 §1 calls it "resolved" | 8 | provenance wording | part 2: `git status --porcelain | wc -l` on the cluster checkout at ff230621; re-word the stamp "tracked tree clean" |
| F6 | B8.2 design note: mandatory-total and wall-time lines inconsistent with its own table (125–471 vs 130–475; 8.9–33.7 h vs 13–46 h) | 14 | arithmetic consistency | append reconciliation before S4 registration |
| F7 | Item 19's cluster cost primitives have no local `sacct` copy | 19 | F4 provenance | part 2 re-pull; save the dump under `wave2_20260829/` |
| F8 | KW-Q1 record does not carry the E21 narrowing of "form-invariant" (exact for the global normalisers, first-order for the divisor gap) | 6 | completeness | append to `B4_2_KWQ1_RUN_FORM_NOTE.md` |
| F9 | **This pass's own deviation**: items 12–19 were verified and adjudicated by the same agent (payload gap, disclosure 1); the dedup memo covers items 1–11 only | — | process | author may order a supplementary sonnet panel on 12–19 |
| F10 | The brief's HEAD (`b87ad2e6`) was one commit behind the tree (`85dae577`); no verdict affected | — | cosmetic | — |

Rule-2 (builder ≠ runner) audit across the 19 items: satisfied where a registered measurement
ran (items 2, 3, 6, 8, 11, 15 — builder, runner and reader distinct and named); disclosed as
conflated on code-correctness gates without a runner role (items 7, 9, 12, 13); unverifiable
from artefacts for item 10 (F3). Rule-1 (append-only): verified by `git diff`/`git show` on every
touched record cited in items 1, 3, 4, 5, 8, 9, 10, 11 — pure insertions in all cases;
`git status --porcelain` shows no modified tracked record at adjudication time.

---

## 4. Items that return to the author

Tags per CLAUDE.md: **[RULE]** = a scientific ruling on evidence in front of the author
("ratified" grants it); **[DO]** = authorise work ("approved" grants it); **[STANDING]** =
pre-authorise a class of future decisions (granted only when said explicitly). None of the
items below is covered by rows #221–#223: every one has inputs that post-date the grant. Each
carries its inputs, the exact one-line question, the docket-2 §6 cross-reference, and which
verifier items raised it.

### 4.1 [RULE] / [DO] / [STANDING] items

| # | tag | question (one line) | inputs (with provenance) | docket 2 §6 | raised by |
|---|---|---|---|---|---|
| A1 | [RULE] | Mass-window design: (a) adopt log k=3 as a documented structural design choice, (b) keep linear k=1.5 and document the one-sided geometry as deliberate, or (c) commission a k-scan / mass-law-keyed window first — and is a joint_r1 arm required before any word? | Δmean_h,pred=+0.003523 INTERMEDIATE (0.003 < x < 0.008), sign up; true-host retention 66/76 both arms (0/76 changed); 621/1588 collapse all dark-class; mirror 78.9 % retention = linear-Gaussian-law artefact (pull read, CV median 1.018); R6 1D untouched; C3 cost 4.97 CPU-h (`b5_2_readout.json`, `b5_pull_read.json`, row #247). **Folds in the still-open WGEOM §9 item 2 F-ii consequence ruling** (`PREREGISTRATION_MKER_WGEOM_20260828.md:245`) — adjudicator's reading: same design object (ε-derived log-symmetric window vs linear), one ruling closes both; the author decides whether they are one object. | item 1 | items 7, 8 |
| A2 | [RULE] ×4 | PA-HIER-31 open items — one word each: (a) R1: for **CoR-P**, is PA-HIER-10's unconditional `smear_sigma_z=True`-at-every-node pin or PA-HIER-31(b)'s `smear_global_selection=False` authoritative? (b) R2′: the same pair for **CoR-M / S0-A** (the P0 run of record used the narrowed form; its STOP is scoped to sites 2.1/2.2). (c) Ratify the forensic's zero-compute diagnosis of the 5.718e-4 informal-parity residual (generator grid 401→4001, E19) as its disposition and retire the "one re-run of the bank" step — or order the bank regenerated at the current grid? (d) Registration amendments: replace the ±ln√2 secant (intrinsic +0.0455/event bias, E13) and drop the z-binned θ read (selection artefact, E16)? | F-A: `combined_no_bh` 7.447e-3, α_G^φ −12.018 %, D̃_φ −0.745 % (item 16, exact); E19/E13/E16 in `B1_1_S0A_DEFECT_FORENSIC_20260829.md`; the driver still runs ±0.02 vs PA-HIER-29's ±0.0661 (item 1). Resolving measurements for (b): mirror P1′ + CoR-P P1′, 0.33 CPU-h each, registered, unexecuted (R4′). | item 2 | items 1, 2, 16 |
| A3 | [DO] + [RULE] | Authorise the gate presentation for a θ-consistent no-BH divisor Σ^φ(θ) (site 2.3 extended to the phi-table branch, byte-identical at θ=(0,1)) plus a sky-cone-radius flag (`bayesian_statistics.py:4869`, hardcoded 1.5) as the first node of the next tree? | E10/E11: C_b −2.20/−2.25, corrected Z_b −3.68 → −0.62 (chair-reproduced); E12 Z_s → −0.5±1 with the cone flag; E20 edge case (0.075 % of pool rows); cost 35–60 min wall (b) + ≈1.5 h wall (s), UNMEASURED; trigger file. B1 is STOPPED, so this is a new node, not covered by row #223. | item 3 | items 2, 5, 6 |
| A4 | [RULE] | After the wave-3 blind readout + the C0′ off-gate land: ratify `catalogue_numerator_survival_2d="mz_sel"`, center `"eff"` as the production default, or revert to `"off"` pending falsifier (ii)? | `d4765539` (item 12: confined to the declared sites, 12/12 pins, suite 1896/15/27, five scripts pinned); C4 +0.0025057 IMMATERIAL-PREDICTED (item 11); falsifier (i) PASS; falsifier (ii) UNRUN (row #220; ≈40–60 CPU-h at the corrected anchor); ×2.25–2.35 identity residual (row #211) disclosed; row #253 now filed; the A14 read (T_mat=0.008, both venues) not yet made. Pre-authorised to be *decided* under row #223; the ratification of the flip is this fresh [RULE]. | item 4 | items 6, 10, 11, 12 |
| A5 | [RULE] | Re-grade G7 row 16 from "MEASURED, calibration-affecting" to "mock: zero by construction; real data: O(1) degeneracy with the population's z-evolution, hierarchical marginalisation required", and retire rows #137/#138 as citations? | §F provenance (draw law byte-identical at `03cfe80:328`, item 4); HEAD dark-class score −0.4668/−0.3938 vs STALE −0.635/−0.565 (7.16σ/5.95σ); real-data sensitivity −0.60 on bins 2–5 per M1-shape swap. | item 5 | item 4 |
| A6 | [RULE] | S0-B (C1): launch only after the divisor fix (A3), or launch now as a REPORTED-ONLY read with the post-hoc ρ(θ) subtraction disclosed — one word, launch-after-fix / launch-now? | forensic §7: predicted by-construction S0-B non-null ≈ ⟨c⟩·C_b ≈ −1.3 per unit b on production; C1 re-costed ≈7–27 CPU-h; P6 CLI landed (`fb9d8aff`); A2(a)/(b) open. | item 6 | items 1, 2 |
| A7 | [RULE] | [CMEM] pooled-observation awareness: two independent fleets read deficit-direction (p=0.0152 row #219; p=0.0358 A1, 68 % power at −16 %); no pooled statistic was computed or banked (confirmed). Leave unregistered, or open a registration? | item 3 (bit-identical re-execution); `EXONERATION_REGISTER` and DO-NOT-RE-TRY grep clean for this mechanism. | item 7 | item 3 |
| A8 | [RULE] | R2c: bank-and-park, or a ≥90 %-power follow-up (≈3× strata ≈ 30 mirror seeds × 2 arms, ≈15 CPU-h local, chair estimate) — row #220's "one word still required"? Adjudicator: **not moot** — A1's non-trigger of A2 answered the trigger question, not this standing word. | as A7 | item 8 | item 3, registration §3 |
| A9 | [RULE] | B6.1 judgment call: confirm the implemented reading (σ_z,pv from the raw, unshifted host z — the appended note's prose and prereg §1.2's `z_centre`) over the note's §2 formula literal (z̃)? | item 9: hooked call matches the raw-z form at rtol 1e-9 and does not match the z̃ form; bit-identical today (`SIGMA_V_PEC_KM_S=0.0`). | not in §6 | item 9 |
| A10 | [RULE] | Does the per-candidate p_Di instrumentation hook needed for B4.3's instrumented run (3.4 CPU-h; serialises candidate z/mass/weight/is_true_host) count as a physics-trigger change requiring the full `/physics-change` gate, or the lighter instrumentation guard? | `B4_1_IMP_DECOMPOSITION.md` §7; row #249 names it "contingent on a non-physics-hook ruling"; touches `bayesian_statistics.py`. | narrower than item 3 | items 5, 6 |
| A11 | [RULE] | Row #167 (open since 2026-08-22, no later row): for the impostor-weight-switch family, does D̃_φ also complete (COMPLETED-MATERIAL +0.0344) or not (COMPLETED-SMALL −0.00281±0.00047)? This bounds C1's remedy-family range [0, +0.123] in the [IMP] claim card. | `BIAS_HISTORY_LEDGER.md:2417`; `CLAIM_IMPOSTOR_DRAG_20260829.md` C1. | not in §6 | item 5 |
| A12 | [RULE] | Should `catalogue_numerator_survival_2d_center="auto"` (not in code) be implemented, and does that implementation need its own gate presentation before code? | proposal §8 item G-3; trigger file. | not in §6 | item 10 |
| A13 | [DO] | Archive the local-only registered-run artefacts now: either `git add -f` the COMMIT_PLAN_3 slice (41 files, 6.4 MB — 33 per-node `event_likelihoods.csv` ≤1.28 MB each + 8 logs) or add `hier_s0_registered_run/`, `kwq1_registered_run/`, `kwq1_parity_run/`, `b1_1_forensic_work/` (1.8 GB total) to the Option-A archive; and run `archive_run_wave2.sh` on the wave-2 out-roots once SSH returns (expiry 2026-09-23). git-force-add / archive-only / both? | F4 (§3.2). | not in §6 | item 18 |
| A14 | [DO] | Zero-compute housekeeping bundle (append-only, no physics): replace the `(row #<adoption>)` placeholder at `bayesian_statistics.py:3274` with row #253 (comment only); append the log-text reconciliation to the B7.3 presentation §13.1 / verifier report; append the citation fixes F1/F2/F6/F8; file the one-line GitHub-rejection note and the 8.6 CPU-h unbanked line in the ledger; re-word the A22 stamps "tracked tree clean". Approve as a batch? | §3 | §7 rank 7 | items 1, 6, 7, 12, 14, 18 |
| A15 | [DO] | Stage P costing: confirm it stays **moot** (not re-costed, not launched) until A3/A6 resolve — given the measured 18.6× under-costing of θ-engaged smeared cells (≈320 CPU-h unsmeared, ≈13.6 h wall per smeared cell)? | item 1; docket 2 §2 B1. | not in §6 (docket §7 "not recommended") | item 1 |
| A16 | [DO] | S0-R: confirm it stays FALLBACK/DISARMED (PA-HIER-28 item 5) and is not scheduled for a future session? | item 1 §3; PA-HIER-28. | not in §6 | item 1 |
| A17 | [STANDING] | The row #222 grant lapses at this verifier pass by its own text. Does the author issue a **new** standing grant for the next tree (docket 2 §7 candidates, ranked: divisor fix + S0-A re-certification + S0-B; B4.3; B8.2 S1–S5; B7 falsifier (ii) + `off` arm; k-scan; CMEM follow-up), with what scope (instruments / arms / registrations / path choices / production-default flips) and what lapse condition? Nothing in the next tree runs on the old grant. | rows #222/#223 (verbatim, `BIAS_HISTORY_LEDGER.md:3018-3020`); this report. | §7 | adjudicator |

Reconciliation with registration §3's seeded minimum set: F-ii (A1) ✓; PA-HIER-31 open items
(A2) ✓ — with the correction that author item 3 is *not* among them; B7.3 ratification (A4) ✓;
G7 row 16 (A5) ✓; S0-B question (A6) ✓; CMEM pooled note (A7) ✓; R2c word (A8) ✓ — **still
open, not moot**; WGEOM §9 F-ii — **untouched by any node as a separate ruling**, folded into
A1 by docket 2 and by this pass. Docket 2 §6 items 1–8 map onto A1–A8 one-to-one; A9–A17 are
additional (five from the verifier panel via the dedup memo, two from this pass: A13, A17).

### 4.2 Orchestrator path decisions of record (taken under row #222 judgement; open to VETO)

Each was taken on results that existed at the time, is recorded verbatim in the ledger, and
is what a veto would unwind. None is an approval request.

| # | decision | ledger | consequence if vetoed |
|---|---|---|---|
| P1 | B1 → S0-B proceeds only after PA-HIER-31 + P6 + P0, in the `"2.2"`/unsmeared CoR-P-faithful form (site 2.3 OUT OF SCOPE by F-A); then, on B0-A′, **B1 STOPS at 1.1**, C1 not submitted, forensic node opened | #239, #250 | C1 template exists; a veto = A6 launch-now |
| P2 | B2 PARKED with the bound (A2 not triggered) | #239, #226 | veto = A7/A8 follow-up |
| P3 | B3 CLOSED as PREMISE-REFUTED; C2 STRUCK; L1/L4 re-cut | #239, #240 | veto = re-open a population-prior instrument (no premise found by items 4) |
| P4 | B4 → KW-Q1 now in the `"2.2"`/unsmeared form (algebraic form-invariance); then → 4.3 derivation path, **no merge into B1** | #239, #249 | veto = merge clause / different mechanism path |
| P5 | B5 → C3 as registered; INTERMEDIATE banked REPORTED; no k=3 in wave 3; F-ii returns (A1); L10 to B8.2 | #239, #247 | veto = adopt or k-scan before the word |
| P6 | B6 CLOSED at depth 1 (`1f003da6`, C0-certified) | #239 | — |
| P7 | B7 → C4; on IMMATERIAL-PREDICTED, **7.3 adoption gate opened and executed** (`d4765539`), batched as the ONLY adoption into the wave-3 blind readout (F2) | #248, #253 | veto = revert to `"off"` before wave 3 (A4 decides after) |
| P8 | B8 → harness stages S1–S3 local, overlapping wave 2 (designed, not built) | #239 | — |
| P9 | Wave-2 cluster set = C0 + C3 + C4 first, C1 held; wave-3 set = C0′ off-gate + 2×41-task blind arrays (built, DRY_RUN, not submitted) | #245, #252, `85dae577` | veto = different wave-3 composition |
| P10 | Deviations from docket 1 §2: B1 three corrections (unsmeared form, P6, P0 re-scope), strike C2, batch 13 tasks/179–357 not 16/224–447 | #238, #239 | confirmed argued with numbers (item 17) |

---

## 5. Deferred item 20 — the wave-3 blind HEAD readout (part 2 appends here)

**State at adjudication:** wave 3 is fully built and NOT submitted. `cluster/submit_wave3.sh`
(`DRY_RUN=1`) prints, in order: (0) `wave3_c0prime_off_gate.sbatch` — 2 tasks (iiib, joint_r1),
h=0.730 only, explicit `--catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset`;
(1) `wave3_headreadout_iiib.sbatch` — 41 tasks, `H_GRID_41`; (2) `wave3_headreadout_joint_r1.sbatch`
— 41 tasks. Estimated 159.8–290.1 CPU-h (blind arrays) + ≈6 CPU-h (C0′). Blocked on SSH
(`Permission denied (publickey,keyboard-interactive)` under `BatchMode=yes`; only the author
can re-authenticate).

**What part 2 will check (registration item 20, plus the verifier-scope delta from docket 2):**

1. **F2 honoured.** B7.3's `mz_sel`/`eff` flip is the ONLY production-default change riding the
   readout. Verified now at the script level: neither blind sbatch passes a 2D flag
   (`wave3_headreadout_iiib.sbatch:13-19,158-159`; joint_r1 `:19-27`) — the default resolves
   to `mz_sel`/`eff` at `d4765539`; both pass the explicit post-wave-2 identity values
   `--mass_filter_geometry linear --mass_filter_k 1.5 --theta_b 0.0 --theta_s 1.0 --theta_sites all`.
   Part 2 re-checks the *submitted* command lines and the provenance JSONs (commit hash =
   wave-3 commit; `tree_dirty_file_count` explained).
2. **The T_mat = 0.008 falsifier (A14).** |Δmean_h| on the 2D channel, BOTH venues, against the
   pre-adoption baseline; ≥ 0.008 on either venue falsifies IMMATERIAL-PREDICTED (the registered
   C4 prediction is +0.0025 on iiib, direction up from the HEAD 2D offset −0.066653 iiib /
   −0.066987 joint_r1); R1 (eventwise inequality) and R6 (1D bit-identical) at full grid —
   any violation = INSTRUMENT-DEFECT and reverts the default to `"off"`. The same T_mat=0.008
   (`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`, row #213) used in items 8, 11, 12 — part 2
   confirms no other threshold is inferred.
3. **The C0′ off-gate baseline logic** (`cluster/WAVE3_SUBMISSION_NOTE_20260830.md` §1a,
   `85dae577`). Purpose: the A14 read needs a pre-adoption baseline; a full 82-task `off` array
   would cost another ≈160–290 CPU-h. C0′ instead runs explicit `off`/`unset` at the wave-3
   commit at h=0.730 on both venues and diffs against the banked 2026-08-27 readouts
   (`headreadout_20260827/{iiib,joint_r1}`, commit `d04d9dc9`): band ≤1e-12 relative on the 14
   diagnostic columns + md5-identical `posteriors/h_0_73.json` and
   `posteriors_with_bh_mass/h_0_73.json` (the C0 gate's own form, which PASSED at 0.0 —
   item 15). **PASS ⇒ the banked readouts ARE the pre-adoption baseline** and the blind delta
   is read against them directly. **FAIL ⇒ the 82-task `off` array becomes necessary, and the
   per-column diff at h=0.730 is diagnosed first** (which commit moved behaviour at the flag's
   own pre-adoption value) before launching it. Adjudicator's two cautions for part 2: (i) C0′
   certifies one h-node; the A14 read spans 41 nodes — the single-h economy is an inherited
   assumption (C0's, which held), to be stated as such in the readout; (ii) joint_r1 has no
   registered C4 arm, so its A14 read is a first read against a prediction transferred from
   iiib, not a falsification of a per-venue registered number — the readout must say so.
4. **Attribution discipline (F3/F2).** With exactly one adoption riding, the blind delta and
   the per-change delta coincide by construction; the readout may *report* the delta as the
   A14 test of the registered C4 prediction but may not *attribute* any other part of the
   HEAD move to B7.3 or to anything else — per-change attribution stays with the registered
   arms (C4; the unrun falsifier (ii)).
5. **Falsifier (ii) status** (row #220, unrun) must be carried PROVISIONAL in whatever claim
   the readout supports; the readout does not discharge it.
6. **Carried from part 1:** re-pull `sacct` for jobs 6738998/6738999/6739000/6739001 and save
   the dump (closes item 19); retrieve C4's provenance extras (closes item 11's PROVISIONAL on
   provenance); confirm the 296 untracked files on the cluster checkout (F5); run the wave-2
   archive script (A13); register the wave-3 datasets (`datasets.yaml`, `DATA_INVENTORY.md`).
7. **Stop/continue framing, stated in advance** (docket 2 §3): the B8.1 centering clause
   (|⟨h⟩−0.73| ≤ 3σ_floor = 0.0052) is already known to FAIL — HEAD 2D bias −0.0668 is 38
   floor-σ and the one change riding wave 3 is predicted to move it by +0.0025. Wave 3 is the
   A14 falsifier and the G41 baseline for the next tree; it is not a stop verdict, and part 2
   will not read it as one.

**Part 2 of this report will be appended below this section, under its own dated stamp,
after the readout lands; nothing above this line is edited.**

---

## 6. Plain-language summary (≈250 words)

The root goal is an inference set-up that is unbiased up to the level where the information
has genuinely run out. Fan-out 1 moved that goal in three ways.

**Established.** The floor is now a number: with GLADE photo-z at the production N=1588 the
single-host Fisher floor is σ_h = 0.00175 (0.24 % of h), and the with-BH-mass channel adds
nothing at any realistic mass scatter (item 13). The measured 2D width is ≈11× that floor and
its centre ≈38 floor-σ off — so the remaining bias is an estimator-consistency budget, not
starvation. The mock's dark-host population prior is the generator's own law (item 4), so the
dark-class tilt has no population explanation left; it is a selection/impostor object. The
impostor leg is a necessary cause of the 1D rail (removing it un-rails 1D to 0.713), its
kernel width is inert (R = +0.085), and it localises to z < 0.36 (items 5, 6). One production
change landed, confined to its declared sites and predicted immaterial (+0.0025) by a
registered arm (items 11, 12).

**Refuted.** The mirror's "3σ window keeps 99.7 % of hosts" transfers nowhere (production
loses 0/76 hosts; the collapse is 621 dark-class events); the "site 2.3 is inert" headline (it
reaches the no-BH read through α_G^φ); every wave-2 cluster cost estimate (9–28× high); and the
[HIER] instrument itself — its no-BH divisor carries no θ, so the score-at-truth null fails by
construction (Z_b −3.68 → −0.62 once restored).

**Undetermined.** The net H₀ sign of the k=3 window (INTERMEDIATE, +0.0035); the +0.11-high
pure-completion leg; the production θ-score (never measured); the width factor F; and, in
this pass, the cluster `sacct` primitives behind 13.5 CPU-h (item 19) and the wave-3 blind
readout itself (item 20, SSH). Seventeen decisions return to the author; none was pre-approved.

---

*Registered verifier pass, part 1, 2026-08-30; author check per row #222. Adjudicator:
inherit-tier, `xhigh`; inputs: 11 sonnet verifier verdicts + item-12 fragment, the dedup memo,
registration §1/§3/§5, dockets 1–2, ledger rows #221–#253, and the eight orphaned verifier
scripts executed in foreground (`verifier_pass/item1{2..9}_rederive*.py`). Append-only.*

---

# PART 2 — ITEM 20 (deferred): the wave-3 blind HEAD readout and its A14 delta read

*Appended 2026-08-31. Append-only: nothing above this line is edited. Verifier: opus, foreground,
read-only (no ssh, no commit, no code edit); ordered by the author, ledger row #278 item 6;
falsification brief A20 (re-derive every decisive number from source; refuted and undetermined are
valued verdicts). Subject: wave3_20260830/WAVE3_A14_DELTA_READ_20260831.md and ledger rows
#281/#283. Scratch and scripts: tree2_20260830/full_verification_20260831/work/item20_*.py.*

## P2.1 Verdict table

| # | claim | verdict |
|---|---|---|
| 1 | C0-prime off-gate PASS, bit-identical, both venues (row #281) | **CONFIRMED** |
| 2a | A14 verdict: NOT MATERIAL, both venues inside T_mat = 0.008 | **CONFIRMED** |
| 2b | the specific numbers in the record (banked mean_h, wave-3 mean_h, Delta) | **REFUTED** — computed with unweighted grid nodes, not the frozen T0 gradient-trapezoid weights |
| 2c | 1D channel exact-zero, both venues | **CONFIRMED** (bit-identical, not merely rounding-identical) |
| 2d | MAP moves: iiib 0.665 to 0.665; joint_r1 0.660 to 0.665 | **CONFIRMED** |
| 3 | STOP-checks: 41 h x 1588 events, zero non-positive, commit 1e092e82, pins OK | **CONFIRMED** |
| 4 | scorer cross-check vs compute_canonical_combined_posterior, tol 1e-6 on discrete_map | **CONFIRMED** (1D channel; see P2.5 for what was and was not run) |
| 5 | registration conformance with PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md section 8 | **CONFIRMED WITH TWO CAVEATS** (P2.6) |
| 20a | F2: the mz_sel/eff flip is the only production-default change riding the readout | **CONFIRMED** |
| 20b | delta compared to the same T_mat = 0.008; no per-change attribution from a blind multi-change delta | **CONFIRMED** (the A14 arm is itself the registered per-change arm) |
| 20c | falsifier (ii) unrun status disclosed as still provisional | **CONFIRMED** |

**Net: the record's A14 PASS STANDS.** The verdict is robust to the scorer error found in P2.3 —
under the frozen convention the deltas grow but stay at 31 and 51 percent of T_mat. The published
numbers, however, are wrong and the record mislabels its own scorer.

## P2.2 Claim 1 — the C0-prime gate (row #281): CONFIRMED

Re-run independently (item20_c0prime.py). Comparison of
wave3_20260830/c0prime_off_{venue}/simulations/diagnostics/event_likelihoods.csv against
headreadout_20260827/{venue}/event_likelihoods.csv, restricted to h = 0.73, 1588 rows each,
event_idx sets element-wise equal after sort:

- **max_abs = 0.000000e+00 on every one of the 14 numeric columns** (w_G, w_G_legacy, w_tilde_G,
  alpha_G_phi, r_Malm, D_tilde_phi, L_cat_no_bh, L_cat_with_bh, B_num, B_num_wbh, g_frac, L_comp,
  combined_no_bh, combined_with_bh), both venues. Claimed max_abs 0.000; re-derived 0.000e+00.
- md5 of the four posterior JSON pairs, all identical:
  iiib/posteriors/h_0_73.json 563ef45b0598dcfc8f5c9ba19efbf9fd;
  iiib/posteriors_with_bh_mass/h_0_73.json 2b4fb3e0d055e08fe7a905b8d3c4d817;
  joint_r1/posteriors/h_0_73.json 681364526966e835696946c4733456bb;
  joint_r1/posteriors_with_bh_mass/h_0_73.json ee0ecb5cb7ad0c9d0bfbf22b9551ca98.

The wave-3 CSV carries three extra columns (den_log_term, num_log_term_no_bh,
num_log_term_with_bh) absent from the banked CSV; these are new diagnostics, outside the
registered 14-column band, and correctly excluded from the comparison.

## P2.3 Claim 2 — the A14 delta: verdict CONFIRMED, numbers REFUTED

The frozen T0 convention is stated at MEASUREMENT_HEAD_READOUT_20260827.md:58-78 and points at
results/prod2d_closure_20260818/bscale_counterfactual_exploratory.py:23-30. I read that reference
implementation directly. It is: ln P(h_k) = sum_i ln L_i(h_k) over events (uniform prior, raw sum
log L); **w_k = np.gradient(h)_k**; p_k proportional to exp(ln P - max ln P) times w_k, normalised;
mean_h = sum p_k h_k; MAP = discrete grid argmax. No floor, no clipping.

H_GRID_41 is **non-uniform** (0.010 step in the wings, 0.005 across the peak 0.655 to 0.785), so
np.gradient weights are NOT constant: 0.010 on nodes 0-4, 0.0075 at node 5, 0.005 on nodes 6-32,
0.0075 at node 33, 0.010 on nodes 34-40. Dropping the weights therefore changes mean_h at the
third decimal. That is exactly what happened.

Re-derived under the frozen convention (item20_a14.py, item20_1d.py):

| venue | channel | banked mean_h | wave-3 mean_h | Delta |
|---|---|---|---|---|
| iiib | 2D | **0.663347** | **0.665854** | **+0.002507** |
| iiib | 1D | 0.605309 | 0.605309 | +0.000000 |
| joint_r1 | 2D | **0.663013** | **0.667127** | **+0.004114** |
| joint_r1 | 1D | 0.611683 | 0.611683 | +0.000000 |

Decisive pairs, claimed vs re-derived: iiib 2D Delta **+0.002127 claimed vs +0.002507
re-derived**; joint_r1 2D Delta **+0.003519 claimed vs +0.004114 re-derived**.

**The mechanism is identified, not merely asserted.** Scoring the same four CSVs with UNIFORM
(all-ones) node weights reproduces every published figure to the digit:
iiib banked 0.666425 (published 0.66643), wave-3 0.668552 (published 0.66855), Delta +0.002127
(published +0.002127); joint_r1 banked 0.666218 (published 0.66622), wave-3 0.669737 (published
0.66974), Delta +0.003519 (published +0.003519); 1D unweighted 0.605322 (published 0.60532) and
0.611888 (published 0.61189). All six 2D figures and both 1D figures match the unweighted variant
exactly and the weighted variant not at all. The record's own phrase "discrete posterior on the
41-grid" is the tell: the grid was treated as if uniform.

**Two independent corroborations that the gradient-weighted numbers are the correct ones.**
(i) The banked column is checkable against the banked record: MEASUREMENT_HEAD_READOUT_20260827.md
section C.1/C.2 (lines 706-707, 723-724) publishes mean_h 0.663347 (iiib 2D), 0.663013 (joint_r1
2D), 0.605309 (iiib 1D), 0.611683 (joint_r1 1D) — identical to my re-derivation to all six
decimals, and NOT equal to the delta read's 0.66643/0.66622/0.60532/0.61189. The delta read's
"banked mean_h" column therefore does not contain the banked row #213 numbers at all, despite the
document asserting "same scorer as the banked row #213 readout" and row #283 asserting "Frozen T0
scorer". (ii) Section 8's own registered REPORTED-ONLY point prediction for iiib is
Delta-mean_h approximately +0.0025. The gradient-weighted re-derivation gives **+0.002507**; the
published unweighted figure gives +0.002127. The registered prediction lands on the corrected
number, not the published one.

**Consequence for the verdict.** T_mat = 0.008. Corrected deltas +0.002507 and +0.004114 are 31.3
and 51.4 percent of T_mat (the record claimed 26.6 and 44.0 percent). Both remain strictly inside
the band and the sign is unchanged (upward, toward truth), so **A14 PASS is unaffected** and no
part of section 8's falsification map is triggered. The finding is a numerical-accuracy and
record-integrity defect, not a verdict reversal. The record's narrative phrase "a quarter to a
half of T_mat" happens to survive the correction ("a third to a half" would be exact).

**1D exact-zero is stronger than reported.** The per-node summed log-likelihood vectors are
bit-identical between arms: max over the 41 nodes of |sum_i ln L_i(wave3) - sum_i ln L_i(banked)|
= **0.000e+00** for combined_no_bh on both venues (for combined_with_bh it is 3.536 on iiib and
4.593 on joint_r1). So the 1D exact-zero is not an artifact of rounding at 5 decimals and is
independent of the weighting bug — it holds under both conventions.

**MAP: CONFIRMED.** Discrete grid argmax, 2D channel: iiib banked 0.665 to wave-3 0.665 (no move);
joint_r1 banked 0.660 to wave-3 0.665 (one grid step, 0.005). Matches the record. 1D MAP 0.600 in
all four cells. MAP is weighting-independent, so this claim is untouched by P2.3's defect.

## P2.4 Claim 3 — STOP-checks: CONFIRMED

Re-derived from the wave-3 CSVs (item20_stop.py), both venues:

- 41 distinct h, matching H_GRID_41 (imported from darksiren_emri.validation.correspondence_1d)
  element-wise; 1588 distinct event_idx; 65108 rows = 41 x 1588 exactly.
- Zero duplicate (event_idx, h) pairs.
- Event-id set equals the banked headreadout set exactly, both venues.
- **Zero non-positive values in combined_no_bh and in combined_with_bh**, both venues, both
  channels (also zero in the banked CSVs). No sentinel signal.
- 41 h_*.json in posteriors/ and 41 in posteriors_with_bh_mass/, both venues. (joint_r1 carries
  one additional file, realization_provenance.json, in each directory — provenance, not a grid
  node; not a defect.)
- Provenance: all 41 run_metadata_*.json per venue carry
  git_commit = 1e092e82a7fea45fd20c23dfdbc2b96e562be322, i.e. **1e092e82**, unanimous, 82/82.
  The key is spelled git_commit, not GIT_COMMIT_AT_RUN as the dispatch put it; same field.
- Dataset pins: 41 of 41 logs per venue carry a "dataset pins OK" line —
  CRB=9a1f2a14384a9281c97ca3be312ddaab, catalogue=c52c13b5cab61f6b3f04bbe202550969, and for
  joint_r1 additionally observed_catalogue=e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751.
  These match the hashes quoted in row #281.

## P2.5 Claim 4 — scorer cross-check: CONFIRMED (1D channel)

**What I ran:** compute_canonical_combined_posterior on
wave3_20260830/iiib/simulations/posteriors/ (the 1D / no-BH channel, 2.0 MB) against the section
1.3 CSV scorer on combined_no_bh from the same run (item20_xcheck.py). **What I did not run:** the
with-BH channel — wave3_20260830/iiib/simulations/posteriors_with_bh_mass/ is 5.1 GB, above what
this foreground pass should load, and the dispatch explicitly permitted the 1D-only substitution.
The 2D cross-check therefore remains **unverified by me** and rests on the run's own section 5
STOP-check.

Result, 1D channel, iiib: combine path n_events_used = 1588, discrete_map = 0.600,
continuous_map = 0.600, mean_h = 0.605309; CSV scorer mean_h = 0.605309, MAP = 0.600.
**|delta discrete_map| = 0.0** (tolerance 1e-6, passes with margin); |delta mean_h| = 2.10e-14;
max absolute difference of the peak-aligned log-posterior shape across the 41 nodes = 1.82e-11.
Both floating-point noise. The two independent code paths agree.

Note that this cross-check is itself evidence for P2.3: the production combine path, run over the
posterior JSONs, yields mean_h 0.605309 — the gradient-weighted value — not the delta read's
0.60532.

## P2.6 Claim 5 — registration conformance: CONFIRMED WITH TWO CAVEATS

Against PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md section 8:

| section 8 requirement | as read | conformance |
|---|---|---|
| threshold: two-sided, band = ratified T_mat = 0.008 | 0.008 cited and applied two-sided | conforms |
| channel: 2D (with-BH) posterior mean_h | combined_with_bh | conforms |
| venues: BOTH iiib and joint_r1 | both read, both inside | conforms |
| grid: H_GRID_41 | 41 nodes, element-wise match | conforms |
| baseline: HEAD default vs HEAD + explicit off | banked 2026-08-27 readout, substituted for the HEAD off-arm under the row #281 C0-prime certification | conforms, with caveat A |
| A22-stamped | stamp set registered at the adoption doc line 519 | not re-verified here (out of the dispatch's five claims) |
| scorer | frozen T0 convention | **MISMATCH — see P2.3** |

**Caveat A (scope of the baseline certification).** Section 8 registers a per-change arm at HEAD.
The C0-prime gate that licenses using the banked readout in place of that arm was run at
**h = 0.73 only** — I confirmed the c0prime_off CSVs contain exactly 1 distinct h value (0.730),
1588 events. The A14 delta, by contrast, is a functional of all 41 nodes. The substitution is
therefore certified at one node out of forty-one. This is the C0-prime gate's registered design
(REGISTRATION_C0_BASELINE_GATE_20260829.md section 14), not a deviation from it, and the 1D
channel's bit-identical sum-log-L across all 41 nodes (P2.3) is strong corroboration that the
off-arm is unchanged grid-wide; but the record should say "certified at h = 0.73, corroborated
grid-wide by the 1D leg" rather than "baseline certified" unqualified. **Undetermined, disclosed —
not refuted.**

**Caveat B (the scorer mismatch of P2.3).** The read does not follow the frozen T0 convention the
registration names. Quoting the mismatch: the record says *"the frozen T0 CSV convention (Sigma ln
combined over events per h; discrete posterior on the 41-grid) — same scorer as the banked row
#213 readout"* (WAVE3_A14_DELTA_READ_20260831.md), and row #283 says *"Frozen T0 scorer vs the
C0-prime-certified banked baseline"*. The frozen convention is *"w_k = np.gradient(h)_k
(gradient-trapezoid weights, P7-2a)"* (MEASUREMENT_HEAD_READOUT_20260827.md:71-ish, section 1.3).
The read used unit weights. Because the delta read's "banked" column was recomputed with the same
wrong weights rather than copied from the banked record, the error is partially self-cancelling
and the verdict survives; had the read instead differenced its wave-3 unweighted 0.66855 against
the banked record's published 0.663347, it would have reported a spurious +0.005 delta.

## P2.7 Item 20's own three checks (a), (b), (c)

**(a) F2 — serialized adoption. CONFIRMED.** Diffing the source tree between the banked readout's
commit (d04d9dc9, from headreadout_20260827/iiib/run_metadata_21.json) and the wave-3 commit
(1e092e82): in darksiren_emri/arguments.py exactly **two** default values are *changed*, and they
are the B7.3 pair — catalogue_numerator_survival_2d from "off" to "mz_sel", and its center from
"unset" to "eff". Every other default appearing in that diff is on a *newly added* flag
(mass_filter_geometry="linear", mass_filter_k=1.5, theta_b=0.0, theta_s=1.0, theta_sites="all",
theta_phi_divisor="off", sky_cone_k=1.5, theta_zwindow="off", z_window_k=1.0,
catalogue_leg_1d_mass_aware="off", candidate_dump_dir=None) — identity-or-off defaults, each
introduced by a [PHYSICS] commit that declared itself byte-identical at its default. That
declaration is not merely trusted here: the C0-prime gate is the empirical proof, since running
HEAD with the one flag forced back to "off" reproduces the pre-adoption readout **bit-exactly**
across all 14 columns and all four posterior JSONs. No second production-default change rides this
readout. **No F2 violation found.**

**(b) T_mat and the attribution prohibition. CONFIRMED.** The same T_mat = 0.008 of items 8/11/12
is used, sourced to MEASUREMENT_HEAD_READOUT_20260827.md:268-285 and row #213. On the F3/F2
prohibition ("any per-change attribution comes only from registered arms, never from the readout's
delta"): the delta read *does* attribute its delta to the single production change, but this is
**not** the prohibited move, because section 8 registers the A14 arm as itself the per-change arm
(HEAD default vs HEAD-plus-explicit-off) and the C0-prime PASS makes the banked readout a valid
stand-in for that arm's off leg. The readout is a one-change delta by construction, not a blind
multi-change delta being retro-attributed. The strength of that construction is exactly caveat A's
one-node certification, which is where a future challenge would land.

**(c) Falsifier (ii). CONFIRMED as disclosed.** The record states, unprompted, that ratification
*"remains pending falsifier (ii) (class-G fleet Option A-prime rung 1, not yet run)"* and that A4
therefore *"RETURNS to the author with these numbers rather than auto-ratifying"*. Row #283 repeats
it. Still-unrun status is explicitly disclosed as provisional; not discharged, and not claimed to
be.

## P2.8 Findings returned to the author

1. **[RULE] Correct the published A14 numbers.** WAVE3_A14_DELTA_READ_20260831.md and ledger row
   #283 carry mean_h and Delta figures computed with unit grid weights instead of the registered
   np.gradient weights on a non-uniform grid. Corrected: iiib 2D 0.663347 to 0.665854,
   Delta +0.002507; joint_r1 2D 0.663013 to 0.667127, Delta +0.004114; 1D 0.605309 and 0.611683,
   Delta exactly 0. **A14 PASS is unchanged.** An append-only correction note is the appropriate
   repair; the verdict does not need re-adjudication.
2. **[RULE] The "banked mean_h" column is mislabelled.** It is not the banked row #213 readout's
   published mean_h (0.663347 / 0.663013 / 0.605309 / 0.611683, MEASUREMENT_HEAD_READOUT
   section C.1-C.2) but a re-score of the banked CSV under the wrong weights. Any downstream reader
   diffing the two documents will see an unexplained 0.003 discrepancy in the *baseline*.
3. **[DO, cheap] Freeze the scorer as code, not as prose.** Three independent readouts have now
   restated the T0 convention in English and one has drifted from it. A single importable
   score_t0(csv, channel) helper, with the banked section C.1 numbers as a regression test, removes
   this failure mode permanently. Cost: under an hour, no physics change.
4. **[Disclosure] Baseline certification is one-node.** Caveat A. Recommend the record's wording be
   qualified; optionally, a cheap grid-wide corroboration already exists (the 1D bit-identity) and
   should be cited in place of a new 82-task off-array.
5. **[Not verified here] The 2D scorer-vs-combine cross-check** (5.1 GB posteriors_with_bh_mass)
   was not re-run. The 1D leg passes at 0.0 on discrete_map; the 2D leg rests on the run's own
   STOP-check.

---

*Item 20 discharged. 10 confirmed, 1 confirmed-with-caveats, 1 refuted (the delta read's published
numbers), 0 verdict reversals. **The record's A14 PASS stands.** Read-only pass; no file above the
Part 2 divider was edited, no commit made, no code changed. Scripts: item20_c0prime.py,
item20_a14.py, item20_variants.py, item20_1d.py, item20_stop.py, item20_xcheck.py under
tree2_20260830/full_verification_20260831/work/.*
