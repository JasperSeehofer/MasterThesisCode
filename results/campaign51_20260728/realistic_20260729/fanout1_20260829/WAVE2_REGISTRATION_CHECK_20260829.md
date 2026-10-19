# WAVE-2 REGISTRATION COMPLETENESS CHECK — fan-out 1 (2026-08-29)

**Launched under rows #222/#223 — charter node: wave-2 PREP chair (registration check).**
**Purpose: INFORMATION for the orchestrator (row #222 form (ii)). Nothing here is an approval
request; every path choice is the orchestrator's; every item goes to the end-of-fan-out verifier.
Append-only. Every number carries {value; source file:line; date} (A11). HEAD at check time:
`dd63fe0c` (tracked dirty: `docs/gates/PHYSICS-GATE-LEDGER.md` +2 lines, six `fanout1_20260829/`
records incl. `hier_s0_driver.py` 567+/42−; untracked
`darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py`) — `git status`, 2026-08-29.**

Inputs read end-to-end: `SYNTHESIS_DOCKET_1_20260829.md` §1–§7 (L1–L9, §4.2–4.3),
`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` (§F, §1–§14 + appended note), `B3_2_POP_FLAG_RECORD.md`,
`PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md`, `B5_2_PULL_READ_20260829.md`, `B7_2_FALSIFIER_I_RECORD.md`,
`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6–§8, §13, `B8_2_HARNESS_DESIGN_20260829.md` §0, §4–§8,
`PREREGISTRATION_HIER_HTHETA_20260826.md` §1.2, §2.1, §2.3, §3.3–3.4, §4.1, §5–§7, PA-HIER-27..30,
`B1_1_HIER_RECORD.md` §1–§2, §5–§6 + appended corrections, `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3,
`COMPUTE_LEDGER.md`, `COMMIT_PLAN.md`, `docs/RESEARCH_CYCLE.md` stage 2 / A8–A15 / A22 / F1–F5,
ledger rows #201, #221–#233, `docs/gates/PHYSICS-GATE-LEDGER.md` tail.

## 0. Chair re-derivations performed (foreground, local, zero `evaluate()` calls, < 2 CPU-min)

Three findings that change the checklist below; each reproducible from banked CSVs in this directory.

**F-A — The docket's P1 premise is REFUTED-IN-PART.** Docket §2 B1 P1 predicted that
`theta_sites="2.2"` unsmeared and `theta_sites="all"` smeared give **bit-identical `combined_no_bh`**
("site 2.3 inert for the no-BH channel under `phi`", B1.1 record finding 4, chair-confirmed in
docket §0(c)). Zero-compute read on seed 900101, the 9 events shared by the P1 smoke node
(`hier_s0_work/b1_2_smoke/p1_2p2_off/s0a_seed900101/node_b_plus_sites2.2_nosmear/…/event_likelihoods.csv`,
event_cap=12, `theta_sites="2.2"`, `smear off`) and the registered b_plus node
(`hier_s0_registered_run/s0a_seed900101/node_b_plus/…/event_likelihoods.csv`, `"all"`, smeared), both at
b = +0.02, h = 0.73 {2026-08-29}:

| column | 2.2/unsmeared vs all/smeared (9 shared events) | verdict |
|---|---|---|
| `L_cat_no_bh` | **bit-identical** (max_rel 0.0) | P1 CONFIRMED for the catalogue kernel: sites 2.1/2.3 add nothing to `L_cat` |
| `B_num`, `B_num_wbh`, `L_comp`, `g_frac`, `w_G_legacy` | bit-identical | — |
| `alpha_G_phi` | 5.8688310e7 vs **5.1635200e7 (−12.0 %)** | NOT inert |
| `D_tilde_phi` | 9.470921e8 vs **9.40039e8 (−0.745 %)** | NOT inert |
| `w_G` = `w_tilde_G` | 0.06196684 vs 0.05492879 | NOT inert |
| `combined_no_bh` | max_rel **7.45e-3** | P1 REFUTED for the combined no-BH likelihood |

Mechanism (source read): the ternary at `bayesian_statistics.py:5187-5191` does pick the θ-inert
`_global_cat_selection_phi` for `global_denom_no_bh`, **but** the path-(A) assembly
`path_a_mixture_objects` (`:2440-2500`; `alpha_G_phi = Σ^4D / n̂_w^φ`, `D̃^φ = α_G^φ + β_Ḡ^φ`, `:2489`)
takes `sigma_4d = _global_cat_denom_with_bh[h]` (`:4160-4171` under `theta_b=_theta_b_23`,
`smear_sigma_z=smear_global_selection`), and the no-BH per-event likelihood is
`(β_G^φ·L_cat + B_num^φ)/D̃^φ` (`:5770`). So under `"all"`+smeared, site 2.3 reaches the no-BH channel
through Σ^4D → r_Malm → α_G^φ → D̃^φ. Whether the −12 % is θ_b at site 2.3 or the smear switch itself is
**UNDETERMINED** from the banked nodes (no (0,1)-smeared node exists); either way it is absent from
CoR-P (`smear_global_selection=False`, `headreadout_20260827/iiib/run_metadata_21.json:cli_args`).
Consequence for the registered S0-A b_plus node (106 events): the mean per-event Δln`combined_no_bh`
= −0.1118, of which −ln(D̃ ratio) = +0.0075 is a **constant global offset** (every C-C event moves by
exactly ×1.00750, measured on the one L_cat = 0 event) and −0.1193 is the kernel part; a C-C class
scored under `"all"`+smeared would return a non-zero constant score (+0.374 per unit b) with zero
per-event scatter — an infinite Z manufactured by the global table, not a lever. **PA-HIER-31 must
register the CoR-P-faithful form (`theta_sites="2.2"`, `smear_global_selection=False`) for S0-B and for
the S0-A remainder**, and the already-run smeared b_plus node is REPORTED-ONLY / non-CoR-P.

**F-B — The GATE PARITY "batch-order" hypothesis is REFUTED.** The 9-event P1 smoke truth node and the
106-event registered truth node are **bit-identical on all 17 numeric columns** over the 9 shared
events (`node_truth_sites2.2_nosmear` vs `hier_s0_registered_run/…/node_truth`; also identical to
`default_regress/…/node_truth`, `"all"`/auto) {2026-08-29}. Summation order does not depend on N here,
so the 5.718e-4 residual of the driver vs the **banked** bc CSV (`p3_b0_work/bc_900101_work`, B1.1
record §2.2) is not a batch-size effect; the live hypotheses are a code/config delta between that
CSV's commit and HEAD, or a process/thread-count effect in the banked run (both smoke and registered
nodes ran at a 14-core pin) — the docket P2(e)'s "one re-run of the banked bc CSV at the current
commit" remains the deciding read. Bearing on C0: a same-N, same-commit reproduction gate CAN fail on a code
delta, which is exactly what C0 is for (A15 control-capable-of-failing: satisfied by this evidence).

**F-C — θ is not on the production dispatch path.** `BayesianStatistics.evaluate()` accepts
`theta_b`/`theta_s`/`theta_sites` (`bayesian_statistics.py:3555-3561`), but `darksiren_emri/arguments.py`
and `darksiren_emri/main.py` expose **no** `--theta_b/--theta_s/--theta_sites` (grep 2026-08-29: zero
hits; `main.py:1741` `theta_s` is a sky-angle variable). Production runs on the cluster go through
`cluster/evaluate.sbatch` + `EXTRA_EVAL_ARGS` (free-form CLI, `evaluate.sbatch:26-31`;
`MEASUREMENT_HEAD_READOUT_20260827.md:477-511`). `hier_s0_driver.py` (`CONFIG_CHOICES = ("b0i","ft")`,
`:130`) drives only `run_mirror_seed_inprocess`. **C1 cannot be submitted today**: a plumbing commit
(arguments.py + main.py call site → `evaluate()`; byte-identical defaults; the B5.1 pattern, non-physics
files) or a production in-process driver is a missing pre-wave item (P6 below).

Other chair checks: `I_HEAD = 1/0.018366² = 2964.6` ✓; `T_mat = max(0.005, 0.0239/3 = 0.007967) → 0.008`
(`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`) ✓; B3.2 §6.1 arithmetic against
`b3_pop_prediction.json:venues.iiib.bins` (+0.081+0.265 = +0.346 … −0.855+0.697 = −0.158; bins 2–5
−0.612+0.603 = −0.009) ✓; B5 quantiles (`b5_window_count.json`: growth median 0.94855, p95 1.49839,
max 10.0; retention 0.78903; jackknife mean 0.78977, SD 0.04553 → SE 0.0093) ✓; cost arithmetic
C0–C4 ✓ (§3.5); the mass window returns `(hosts_without_BH_mass_filter, hosts_with_BH_mass_filter)`
with the no-BH set `candidate_hosts[redshift_filter_mask]` mass-blind (`handler.py:640-700`), so C3's
R6 premise holds by code structure ✓.

---

## 1. Per-arm checklist

Legend: **PASS** · **GAP** (concrete gap stated) · **n/a**. "Same-commit A22" = names the wave-2
commit hash and the baseline commit; "verifier-scope" = the document itself routes to the end verifier.

### 1.1 C0 — shared baseline gate task (serves C3/C4 [and C2 if run]; L5)

| item | status | detail |
|---|---|---|
| registration exists | **GAP** | No standalone C0 registration. C0 is defined only inside its consumers (WIN-K3 §1 "Baseline B"; proposal §6.2 "Code state"; B3.2 §7 control (i)) with **inconsistent column lists**: C3 names `L_cat_with_bh`/`combined_with_bh`; C4 the same; B3.2 `combined_no_bh`; B8.2 S2(iii) "all 41 h". Required: one C0 note (≤ 1 page, appended to the docket or standalone) stating: venue iiib, h = 0.730 (`evaluate.sbatch --array=21`), the CoR-P CLI verbatim from `run_metadata_21.json:cli_args` **plus explicit** `--mass_filter_geometry linear --mass_filter_k 1.5` (stamps the B5.1 default), wave-2 commit hash, gate = **all 17 numeric columns** of `event_likelihoods.csv` at h = 0.73 vs the banked `d04d9dc9` rows (1588 rows; `headreadout_20260827/iiib/event_likelihoods.csv`, 65 108 data rows = 1588 × 41) to ≤ 1e-12 relative (PROD-A0 form, row #201: ≤ 8.5e-15 over 12 columns), plus `posteriors/h_0_73.json` and `posteriors_with_bh_mass/h_0_73.json`. |
| A8 two-sided + referents | PASS (binary gate) | PASS/FAIL at 1e-12 is a reproduction gate, not a hypothesis band; referent = banked CSV. |
| A10 invariants + blindness | **GAP** | Must say: the gate certifies the **three post-`d04d9dc9` estimator commits** (`d40fe5c8` θ-hook, `1f003da6` s-placement, `0b308828` mass-window flag; `git log d04d9dc9..dd63fe0c -- darksiren_emri/`) at production scale and default values; blind to any defect **shared** by `d04d9dc9` and HEAD (it is a reproduction, not a correctness check). |
| A11 stamps | PASS | commit `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`, 2026-08-27T19:40:20 (`run_metadata_21.json`). |
| A14 falsifier | n/a | (gate) |
| A15 at N | PASS (chair-supplied) | control capable of failing: F-B above (a same-N reproduction differs by 5.7e-4 across a code delta on the mirror); false-pass under a code delta impossible at 1e-12. |
| F3 predictions before run | PASS | prediction = bit-reproduction; secondary: reproduces B3.1's dark-class profile (+0.081, −0.332, −0.562, −0.701, −0.855; `b3_pop_prediction.json:venues.iiib.bins`) to ≤ 1e-6 — doubles as B3's L1 baseline pin. |
| F4 cost + archive | **GAP** | Cost 15–23 CPU-h in `COMPUTE_LEDGER.md:44`; archive cell **"pending"** — must read "yes" before sbatch (F4). **Fallback cost not in the ledger**: on FAIL, C3 and C4 re-run their own baselines at 4 nodes each (+59.7–91.6 and +59.7–81.1 CPU-h; proposal §6.2 "H4 with full baseline re-run 119.4–162.2") — add a conditional row. |
| same-commit A22 | **GAP** | The wave-2 commit does not exist yet (dirty tree, §0). Must be the FIRST wave-2 action; C0's stamp = that hash + dirty-state clean. |
| L-lines | PASS | L5 honoured by C3/C4 text; C0 also = S0-B truth node (docket §4.3) and B8.2 S2(iii) gate artifact. |
| verifier-scope | **GAP** | absent (no document). |

### 1.2 C1 — S0-B production θ-score at truth (B1.2) — registration PA-HIER-31 NOT YET AUTHORED

| item | status | detail |
|---|---|---|
| registration exists | **GAP (blocking)** | PA-HIER-31 unauthored; skeleton with every known item in §2 below. |
| instrument on the production path | **GAP (blocking)** | F-C: no CLI θ flags; add pre-wave item **P6** (plumbing commit or production driver) + a production-scale T-ID = C0. |
| CoR-P fidelity of the θ form | **GAP (blocking)** | F-A: `"all"`+smeared is NOT CoR-P; register `theta_sites="2.2"` + `smear_global_selection=False`; site 2.3 OUT OF SCOPE with reason. Optional **P1′** (one (0,1)-smeared node, ≈ 20 min local, 0.33 CPU-h) to attribute the −12 % α_G^φ shift to θ vs the smear switch — informational, not blocking. |
| A8 two-sided + referents | to write (§2 item 6) | `|Z| ≤ 3` per component is two-sided; B0-M materiality needs the 3-node quadratic fit registered; C-C identity check. |
| A10 + blindness | to write (§2 item 8) | |
| A11 | to write | all inputs listed in §2 with sources. |
| A14 | to write (§2 item 9) | |
| A15 at N = 1588 | to write (§2 item 7) | per-event SD proxy from the mirror (14–17 per unit b, n = 105/9) — disclosed transfer; s-component SD UNMEASURED (S0-A s± not run). |
| F3 predictions before run | to write (§2 item 10) | B1 own · B4 (L2) · B3 (L1, reduced by §F). |
| F4 cost + archive | **GAP** | ledger row C1 (`COMPUTE_LEDGER.md:45`) carries both 60–92 (unsmeared) and 81–113 (smeared); after F-A only the unsmeared form is registrable — strike the smeared band; archive "pending". |
| same-commit A22 | **GAP** | as C0; L8 satisfied (`1f003da6` landed). |
| L-lines | **GAP** | L1 must be re-cut (B3.2 struck, §1.3); L2's driver sha1 `5313c319…` is STALE — current `hier_s0_driver.py` sha1 `9f831b9f7d6b8fed820d547bbe8cd64ff00873e3` (567+/42− vs `dd63fe0c`, the B4.2 FT/`--smear`/`--theta-sites` build); re-pin in the record. |
| verifier-scope | to write | PA-HIER-31 + F-A/F-B findings + the b-node choice → end verifier (docket §5 item 1). |

### 1.3 C2 — M1-prior arm (B3.2)

| item | status | detail |
|---|---|---|
| registration exists | PASS (as an instrument-validation read) | `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §6–§9 is a complete F3 registration of the C2 read. |
| instrument exists | **GAP (blocking, by design)** | No `completion_population_prior` flag in the tree (B3.2 builder declined; gate-ledger row "PRESENTED WITH A STOP … NO CODE"; presentation §13 item 2 "No code under this presentation"). The arm cannot run and, per §F, **should not**: premise refuted at zero compute (production dark hosts drawn from `(1−f)·dVc/dz/(1+z)`, `03cfe80:…/dark_siren_injection.py:328`; byte-identical to the estimator's prior), adoption = [WPOP-TUNING] collision at 60× the bounded misspecification (register item 5, `EXONERATION_REGISTER_20260827.md:382-388`). |
| A8 two-sided + referents | PASS | R bands ±0.10/±0.20 with referent arm C2 vs C0; execution-completeness (all three h-nodes) stated (§6.1). |
| A10 + blindness | PASS | §8 (six invariants with dates; NEVER items 5, 6 named; blindness (a)–(c)). |
| A11 | PASS | §14 provenance table. |
| A14 | PASS | §9 — the zero-compute falsifier (generator provenance) has RUN and fired; charter 3.3's "moves toward 0 by the predicted share" correctly rejected as a branch-referent fault. |
| A15 at N | PASS | §7 (paired deterministic; SEM_r ≤ 0.018 on bins 2–5; per-bin ±0.20 at n ≈ 120; controls (i)/(ii)). |
| F3 before run | PASS | §6.1 per-bin (+0.346 / +0.156 / +0.027 / −0.062 / −0.158 − Δ_D; bins 2–5 −0.009 − Δ_D) and §6.2 pure-completion direction UP (`pure-all mean_h ∈ [0.8396, 0.86]`). |
| F4 cost + archive | **GAP (moot if struck)** | ledger row C2 45–69 CPU-h stale; archive "pending". |
| same-commit A22 | PASS | HEAD `dd63fe0c` named (§0). |
| L-lines | **GAP** | L1 and L4 assume the arm runs; §13 recommends striking B3.2 from L1 — the docket lines must be re-cut by an appended note (F1: a dependency line that no longer has an arm). |
| verifier-scope | PASS | §13 item 4. |
| record hygiene | **GAP** | `B3_1_POP_RECORD.md` §3's interpretation ("accounts for essentially the entire tilt") needs the append-only superseding note the presentation asks for (§13 item 1); only the three refuter must-fix items were appended (record tail, 2026-08-29). |

### 1.4 C3 — log k = 3 counterfactual (B5.2)

| item | status | detail |
|---|---|---|
| registration exists | PASS | `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` (panel clean, 0 rounds). |
| A8 two-sided + referents | PASS, one wording gap | three-way map at 0.003 / 0.008 with MATERIAL-UP/DOWN and referents (arm T vs baseline B, §3); secondary T_mat edge. **GAP (minor):** the map is written on "ΔMAP" but on H4 only Δmean_h,pred (stencil) is measurable at T_mat resolution — state that the wave-2 adjudication is on Δmean_h,pred and ΔMAP is delivered by the wave-3 G41 read (proposal §6.2 pattern). |
| A10 + blindness | PASS | §7 (invariants with rows/dates; Instrument J NEVER disclosed; k_sky = 1.5 (L3); blindness (a)–(d)). |
| A11 | PASS, one inconsistency | §4/§8/§9 carry sources+dates. **GAP (minor):** two bands for one quantity — R1 ±2 pp ([0.769, 0.809]) vs §8 falsifier ±3 SE ([0.762, 0.816]); a production retention in [0.762, 0.769) ∪ (0.809, 0.816] fails R1 but is not "falsified". Reconcile by an appended note (recommend the derived ±3 SE band for both; R1's ±2 pp is asserted). |
| A14 | PASS | §8 items 1–3 (retention-transfer, attribution via R6/R1, stencil validity). |
| A15 at N = 1588 | PASS | §9 (paired deterministic ⇒ materiality bands; floor ≤ 8.5e-15; jackknife SE 0.0093 as a lower bound). |
| F3 before run | PASS | §4 items 1–2 (growth median 0.949 / p95 1.498 / max 10.0; retention 0.789 ± 0.009 on iiib as a cross-fleet prediction). |
| F4 cost + archive | **GAP (procedural)** | 44–137 CPU-h in ledger row C3; §11 requires the archive cell to read "yes" before sbatch — it reads "pending". |
| same-commit A22 | **GAP** | §1 "wave-2 HEAD, after B6.1's and B5.1's commits" — no hash (cannot exist yet); append a launch stamp with the wave-2 hash and the baseline hash `d04d9dc9` (C4 names it; C3 does not). |
| L-lines | PASS | L3 (k_sky invariant), L5 (C0), L8 (`1f003da6` landed), L9 resolved (§5, `555f0186` + `MASS_RELATION_ASSESSMENT.md` dating). |
| verifier-scope | PASS (implicit) | §10 routes adoption through the gate + wave-3 readout; add one line "this registration → end verifier". |
| class-definition note | **GAP (minor)** | §4 item 3 "class migration C-A/C-B → C-C" — say whether class is defined at h = 0.730 only (this arm's zero-compute node) or over all 41 nodes (B3.1's definition, `B3_1_POP_RECORD.md:75-82`); the two differ for events with `L_cat_no_bh > 0` only at some nodes. |
| SLURM sizing | PASS | `--time=03:00:00` vs slowest HEAD task 1:25:52 × p95 growth 1.498 = 2.14 h; 16-event zero→nonzero class smoke first (§11). |

### 1.5 C4 — PROD-CF-2D `mz_sel`/`eff` (B7.2)

| item | status | detail |
|---|---|---|
| registration exists | PASS | proposal §6.2 + appended §13.3 (final form, "confirmed unchanged"). |
| A8 two-sided + referents | PASS | MATERIAL-UP/DOWN at ±0.008, IMMATERIAL ≤ 0.004, AMBIGUOUS between / on validity violation → conditional G27; referent arm T vs baseline B. |
| A10 + blindness | PASS | §7 (S_4D table NEVER re-derived — disclosed; blindness (a)–(e)). |
| A11 | PASS | §12 provenance table; §13.1 result table with file:line + dates. |
| A14 | PASS | falsifier (i) IMPLEMENTED + PASS (rel. dev. 2.6e-16/1.3e-16; coded 1.50/5.67; A15 probe 0.60; `test_survival_2d_homogeneity_falsifier.py`, 52 passed); falsifier (ii) 208–286 CPU-h returns separately (row #220) — attribution remains provisional on (ii), which §13.3 should say in one line (**GAP minor**). |
| A15 at N = 1588 | PASS | §6.2 operating characteristics (paired deterministic; R1 strict ⇒ null excluded by construction). |
| F3 before run | PASS-with-note | no numeric prediction (direction REPORTED-ONLY, §6.2) — C4 is not a shared instrument, so F3 is not triggered; noted for the verifier. |
| F4 cost + archive | **GAP (procedural)** | 59.7–81.1 arm / 74.7–101.4 incl. C0 / ceiling 105/132 (§13.3); ledger row C4 shows "60–105"; archive "pending". |
| same-commit A22 | **GAP** | "wave-2 HEAD" — no hash; baseline `d04d9dc9` named ✓. **Additionally**: the falsifier test file is untracked — it must be in the wave-2 commit or the A22 stamp is not clean. |
| L-lines | PASS | L5. |
| verifier-scope | PASS | §11 decision table ("every item returns to the end verifier"). |
| STEP-2 smoke | PASS-with-note | h = 0.730 task pins the 1.0–1.3× overhead; **GAP (minor)**: register the resubmit rule if the task exceeds `--time=03:00:00` (overhead > 2.1×) — a walltime resubmit is not a band change. |

---

## 2. PA-HIER-31 — skeleton with every item already known (to be appended to `PREREGISTRATION_HIER_HTHETA_20260826.md`; nothing above its divider may change)

**Stamp line:** `PA-HIER-31 (2026-08-29; S0-B registration; launched under rows #222/#223 — charter node B1.2; [FABLE-ORCH])`.

1. **Venue and configuration of record (CoR-P).** Production iiib: CRB `seed61000/prepared_cramer_rao_bounds.csv` md5 `9a1f2a14384a9281c97ca3be312ddaab` (1590 rows, 1588 scored; `MEASUREMENT_HEAD_READOUT_20260827.md:42-43`); reduced catalogue md5 `c52c13b5…`; `EVAL_SEED = 777000`; CLI verbatim from `headreadout_20260827/iiib/run_metadata_21.json:cli_args` (`absolute_marginal` / `volume_deconv` / `fused` / `phi` / `smear_global_selection=False` / `pdet_wbh_z_resolved=False` / `eddington_m=on` / `sigma4d_mass_kernel=point` / `catalogue_numerator_survival_2d=off`) + explicit `--mass_filter_geometry linear --mass_filter_k 1.5`; **h = 0.730 only** (`evaluate.sbatch --array=21`). Cluster: `cpu_il`, 16 cpus/task, `--time=03:00:00`.
2. **θ form (F-A).** `theta_sites = "2.2"` (the batched per-host host-z kernel, `bayesian_statistics.py:7091-7101`, the production dispatch path) and `smear_global_selection = False`. Site 2.1 (scalar twin, `:6418-6429`) is not on the production dispatch path and is registered as present-but-inert; **site 2.3 is OUT OF SCOPE** because CoR-P has no smeared global selection (a θ at 2.3 requires `smear_sigma_z=True`, `:2799-2806`, which changes Σ^4D → α_G^φ → D̃^φ for BOTH channels — measured −12.0 % / −0.745 % on the mirror at b = +0.02, §0 F-A — i.e. a non-CoR-P denominator). Registered identity check: for every C-C event (`L_cat_no_bh == 0` at h = 0.73) `combined_no_bh` is **bit-identical** across all five θ-nodes (θ has no referent there); any deviation ⇒ INSTRUMENT-DEFECT. The S0-A remainder (P0) runs in the SAME form for comparability; the registered smeared b_plus node (`hier_s0_registered_run/…/node_b_plus`) is REPORTED-ONLY, non-CoR-P.
3. **Instrument path (F-C).** Pre-wave item **P6**: expose `--theta_b`, `--theta_s`, `--theta_sites` in `arguments.py` + `main.py` → `evaluate()` (defaults `0.0`, `1.0`, `"all"` byte-identical; `run_metadata_*.json` records them = the A22 stamp) — plumbing commit, non-physics files, ledger note; OR a production in-process driver (then the A22 stamp is the driver's own JSON). GATE T-ID at production scale = **C0** (θ = (0,1) reproduces the banked `d04d9dc9` columns ≤ 1e-12). GATE ENG scored on `L_cat_no_bh`: ≥ 99 % of C-A ∪ C-B events move ≥ 1e-6 relative at each off-truth node (mirror b_plus: 105/105, §0). GATE TABLE-FRESH: one `BayesianStatistics` per node (four separate sbatch tasks guarantee it).
4. **θ nodes (P2(a) decision).** Truth (0, 1) = C0. b-nodes: **±0.033** re-derived from `b_max = 0.0661` (PA-HIER-29, 2×median `REDSHIFT_MEASUREMENT_ERROR/(1+REDSHIFT)` = 0.033038; half-step of a 5-node grid over ±0.0661) — chair's recommendation, matching docket P2(a); the S0-A remainder stays at the as-built ±0.02 with a disclosed "as-built" label (paired within arm; the two arms are never mixed). s-nodes: 1/√2, √2 (log-symmetric, unchanged). Freeze rule of §2.3 applies.
5. **Statistics.** Per event i (both channels, primary = no-BH; ln of `combined_*` as in `hier_s0_driver.py:242-245`): `score_b,i = [lnL_i(+0.033,1) − lnL_i(−0.033,1)]/0.066`; `score_lns,i = [lnL_i(0,√2) − lnL_i(0,1/√2)]/ln 2` (the PA-HIER-4 form; relabels the driver's `score_s`, Z identical, magnitudes now in ln-s units — docket P2(c)). `Z_x = mean(score_x)/SEM(score_x)`. Read **pooled** (N = 1588), **by class** (C-A: `in_catalog = True` & `L_cat_no_bh > 0`, n ≤ 76 {B3.2 §F item 3}; C-B: `in_catalog = False` & `L_cat_no_bh > 0`; C-C: `L_cat_no_bh == 0`, n = 606; C-A ∪ C-B = 982 = 0.6184 × 1588 {`b3_pop_prediction.json:venues.iiib.n_matched`; B8.2 §0}) — class defined at h = 0.73 (this arm's single node), disclosed as differing from B3.1's all-41-node definition — and **by z-bin** on B3.1's registered edges {0.075, 0.392, 0.559, 0.659, 0.753, 1.018} (`b3_pop_prediction.json:registered_bin_edges`), using the CRB `z_true` (`dist_to_redshift(d_L, 0.73)`). Curvature leg (B0-M/B0-P): quadratic fit through the three b-nodes → `b̂ = −S′/S″`, `σ_b = 1/√(−S″)`; likewise in ln s.
6. **Bands (A8, two-sided, referents = the four θ-nodes vs C0).** B0-B: `|Z_b| ≤ 3` AND `|Z_lns| ≤ 3` pooled ⇒ **LEVER-DEAD-AT-N (production)**; either `> 3` ⇒ **LEVER-LIVE**, then B0-M: MIXED if `|b̂| < 0.0165` (half the b step) or `|ln ŝ| < 0.5·ln√2 = 0.173`; B0-P (power): `σ_b < 0.0661` and `σ_ln s < ln 2`, else UNPOWERED (no DEAD claim). Per-class and per-bin: same `|Z| ≤ 3` bands, REPORTED; C-C: identity (item 2). All verdicts carry the REPORTED-ONLY cap (PA-HIER-28 item 9). Fork mapping (docket §2 B1): DEAD ⇒ 1.3b; LIVE ⇒ 1.3a (Stage P re-costed under L6).
7. **A15 at N = 1588.** Null: `Z ~ N(0,1)` ⇒ `|Z| ≤ 3` false-fail 0.27 % two-sided; power 80 % at `mean = 3.84·SEM`. Per-event SD proxy (mirror, half-secant b at Δb = 0.02, seed 900101): **16.9 per unit b (n = 105 active, `"all"`/smeared, includes the +0.375 global offset) / 14.0 (n = 9, `"2.2"`/unsmeared)** {§0, 2026-08-29} — a **mirror→production transfer, disclosed** (production balls carry ~10³× more candidates); at N_active = 982: SEM ≈ 0.45–0.54 per unit b ⇒ detectable mean score ≈ 1.7–2.1 per unit b (≈ 0.11–0.14 nats per event across the full ±0.033 secant); C-A alone (n ≈ 76): SEM ≈ 1.6–1.9 ⇒ ≈ 6–7 per unit b (weak, REPORTED). **s-component SD: UNMEASURED** (S0-A s± nodes not run) — filled from P0 before sbatch, else the s bands are registered with the b proxy and flagged. Controls capable of failing: C0; the C-C identity; ENG on `L_cat_no_bh`.
8. **A10 invariants (dates) + blindness.** Carry prereg §5.1 items 1–5, 7, 9–13 (with the B5.1/B6.1 audit date 2026-08-29 added to item 3/7) and the CoR-P list of item 1; `smear_global_selection = False` (2026-08-29, this amendment); K1–K4 host-z kernel form `f_k·dVc/dz/(1+z)` (G2b, 2026-08-04; NEVER re-audited against a z-dependent population — B3.2 §8 item 5); `S̄_φ` table (NEVER end-to-end, B3.2 §8 item 6). Invariant 8 (mirror↔production parity) becomes moot for S0-B (it IS production) but the 5.7e-4 banked-CSV residual stays undiagnosed (F-B). **Blindness:** (a) anything acting only through a smeared global selection (site 2.3, out of scope); (b) the production venue has no truth-θ — a non-zero score cannot by itself separate a photo-z kernel misspecification from any other misnormalisation sharing the catalogue leg (the B4 impostor object; that is why the L2 profile prediction is registered); (c) θ's 2-D span (prereg §5.2 item 2); (d) single h; (e) the with-BH channel is secondary and inherits invariant 12 ([P3-MKER] state).
9. **A14 falsifiers.** LIVE attributed to "the host-z kernel" is FALSIFIED if the C-C identity fails (instrument) or if the s-score z-profile is flat within 3σ AND the L2 q1-share prediction fails (then it is a normalisation object — B4.3's class). DEAD is provisional until B0-P passes and the S0-A remainder certifies the instrument in the same form (P0). The mirror LEVER-DEAD falsifier of prereg §6 (S0-B live with mirror dead) stays as registered.
10. **F3 — predicted profiles registered on the shared instrument (before the first sbatch).**
   - **B1 [HIER] own:** no point prediction (the hypothesis is the LIVE/DEAD fork). Sign expectation, REPORTED-ONLY: `score_lns > 0` pooled (the likelihood prefers a wider kernel if the quoted photo-z errors understate realised scatter — F4/row #193 outlier regime); `score_b`: no sign registered.
   - **B4 [IMP] (L2):** on C-A ∪ C-B, the share of `Σ|score_lns,i|` carried by z-bin 1 (0.075–0.392 ≈ the mirror q1 edge 0.3575) is **≥ 0.50** {mirror analogue: q1 carries 91.6 % (ft) / 86.2 % (fc) of the impostor-leg h-score, `b4_imp_stage1_forecast.json:covariates.{ft,fc}.z_true`; q1 mean `s_imp` = −0.798 ± 0.041 (ft)}; **< 0.50 ⇒ C2's localisation does not transfer to production and is WITHDRAWN there** regardless of Z. Coupling to KW-Q1 (P2): OWNS (|R| ≥ 0.5) ⇒ expected `|Z_lns| > 3` on C-A ∪ C-B with the bin-1 concentration; INERT ⇒ expected `|Z_lns| ≤ 3` in bin 1 — REPORTED alongside, not a band (KW-Q1's R is an h-score response to s, not ∂lnL/∂s).
   - **B3 [POP] (L1, reduced by §F):** (i) C-C class θ-score ≡ 0 (identity; the population prior has no θ referent — K1–K4 untouched by any population flag, B3.2 §1); (ii) the C0 truth node reproduces B3.1's dark-class h-score profile (+0.081 / −0.332 / −0.562 / −0.701 / −0.855; bins 2–5 −0.612, n = 484) to ≤ 1e-6 — the baseline pin; (iii) **no** population-term prediction on the θ-score exists (the data carry no such term, §F); B3.2 §6.1's per-bin bands apply only to C2 and only if C2 is run as an instrument.
11. **Cost (F4) and archive.** 4 θ-nodes × 14.93–22.9 CPU-h = **60–92 CPU-h** (unsmeared form; the 81–113 smeared band is withdrawn) + C0 (shared). Archive: MUST-ARCHIVE (Option A); ledger row C1 "archive-scheduled: yes" before sbatch. Deadline 2026-09-23.
12. **A22 / ordering.** Wave-2 commit hash + dirty-state clean at run START; `1f003da6` (B6.1) precedes ✓ (L8); P0 (S0-A remainder in the 2.2/unsmeared form, ≈ 5 CPU-h / 40 min at 5 parallel nodes) and the GATE PARITY disposition (P2(e), with F-B's corrected hypothesis) recorded before S0-B banks; execution-completeness (A8(d)): no class/bin branch adjudicated until all four θ-nodes exist.
13. **Verifier scope.** This amendment, F-A (site-2.3 non-inertness through α_G^φ), F-B, F-C, the b-node choice and the S0-A smeared-node reclassification.

---

## 3. Cross-arm consistency

1. **One commit, one baseline (L5).** Every registration that names a baseline names the banked HEAD readout (C4 by hash `d04d9dc9`; C3 by document; B3.2 by "banked HEAD"); **none names the wave-2 commit** — it cannot exist until the dirty tree (§0) is committed. Consistent, but incomplete: one launch-stamp note (hash + dirty-state) appended to C3, C4, PA-HIER-31 and the C0 note closes it. The three estimator commits between `d04d9dc9` and HEAD are each unit-pinned byte-identical at defaults; C0 is their production-scale pin.
2. **C0 column union.** C2 → `combined_no_bh`, `D_tilde_phi`, `beta_G_phi`; C3 → `L_cat_with_bh`, `combined_with_bh` (+ R6 on `L_cat_no_bh`, `combined_no_bh`); C4 → `L_cat_with_bh`, `combined_with_bh`; B8.2 → all. ⇒ C0 gates all 17 numeric columns (§1.1).
3. **C3's L9 resolution vs the presented gate.** Consistent: the gate (`PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` §1–§2) defines `σ_lnM = BH_MASS_ERROR/BH_MASS` as the R&V15 ln-space budget; the pull-read confirms it from `handler.py:44,1446-1459` (`sigma_int = 0.5527`), dates B8.1's "0.19 current" to `docs/MASS_RELATION_ASSESSMENT.md` (2026-06-30, pre-`555f0186`), and the registration §5 keeps the presented σ PRIMARY while retiring `ε = 2Φ(−k)` as a retention statement (78.9 % carried, not 99.73 %). The gate's §7 second caveat ("net sign UNDETERMINED") is exactly what C3 adjudicates. One residual for the verifier: the gate's GW-side `σ_lnM,z ≈ M_z_sigma/M_z` is numerically a point (median ~1e-8), so the window is host-side only — consistent with the pull-read's factorisation, not a contradiction.
4. **B3.2 vs B1.2 z-bins.** B3.2 §6.1 uses exactly `b3_pop_prediction.json:registered_bin_edges` [0.075, 0.392, 0.559, 0.659, 0.753, 1.018]; PA-HIER-31 (§2 item 5) adopts the same edges. Consistent. **But** the "shared instrument" of L1 is only the C0 truth node + event set + bins: B3.2's registered quantity is the **h-score** under a flag that will not run, S0-B's is the **θ-score**. L1 must be re-cut to items (i)–(iii) of §2 item 10.
5. **Docket §2 B3 (c) vs B3.2 §6.2 (sign).** The docket expected the M1 prior to move the pure-completion arm toward 0.73; B3.2 derives UP (away), registered before any run. No arm runs; the correction is on the record (B3.2 §6.2). Consistent with A8: registered before data.
6. **C1 cost band.** After F-A the registered S0-B form is unsmeared: 60–92 CPU-h; the docket's "81–113 if smeared" describes a non-CoR-P form and should be removed from the ledger as an option (it was priced on P1's now-refuted equivalence going the other way).
7. **Wave-2 cluster total after striking C2 and fixing C1's form:** C0 15–23 + C1 60–92 + C3 44–137 + C4 60–105 = **179–357 CPU-h** (13 tasks), vs the docket's 224–447 (16 tasks); conditional +120–173 CPU-h if C0 FAILS (baseline re-runs for C3/C4).
8. **B8.2 (local, no arm).** Design note complete for a stage-2 registration (bands placeholders, A15 at n_U = 100, A10 with two NEVER items, falsifiers per branch). Its cost is **130–475 CPU-h local** (B8.2 §6) vs the docket's "≈ 6 CPU-h per sweep" — 20–80× — and must enter `COMPUTE_LEDGER.md` as a local row; no deadline exposure. Cross-link consistent: SHARED-FILTER's falsifier cites "the B3.2 M1-prior arm's per-bin prediction (L1/L4)" — after striking C2 that referent is gone; B8.2 S4 must name another (C0's dark-class profile pin or the count audit itself).
9. **KW-Q1 (P2) and S0-B form.** `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3 names `theta_sites="all"` (smeared) as PRIMARY and `"2.2"` as DIAGNOSTIC-NOT-FIX. Chair check of whether F-A contaminates the KW-Q1 statistic: `s_imp,i = s_full,i − s_pure,i` with `full = (β_G^φ L_cat + B_num)/D̃^φ` and `pure = B_num/D̃^φ` (`kwq1_score.py` header; `bayesian_statistics.py:5770`), so `s_imp,i = Δ_h ln[(β_G^φ L_cat,i + B_num,i)/B_num,i]` — **`D̃^φ` and `α_G^φ` cancel identically** and `β_G^φ` is θ/smear-inert (`precompute_phi_selection_integrals`, `:4199-4203`). F-A therefore does NOT reach KW-Q1's statistic; it reaches only `combined_*` reads (posteriors, the S0-B θ-score). The `"all"`/smeared vs `"2.2"`/unsmeared choice for KW-Q1 is a cost (13.7 vs 8.4 CPU-h) and CoR-P-fidelity question for any secondary posterior read, not a statistic-validity one; the card's labelling may stand. Chair recommendation: run `"2.2"`/unsmeared (8.4 CPU-h) and state in the run record that `s_imp` is form-invariant by the cancellation above (L_cat_no_bh is bit-identical between the forms, §0 F-A).

---

## 4. Orchestrator path decisions — docket §2 recommendations vs this chair's independent view

| # | docket recommendation | chair view | why (numbers with provenance) |
|---|---|---|---|
| 1 | B1: S0-B YES, sequenced P0 → P1 → P2 → P3 | **AGREE, with three deviations** | (a) P1's equivalence is refuted-in-part (§0 F-A: `L_cat_no_bh` identical, `combined_no_bh` max_rel 7.45e-3 via α_G^φ −12.0 %); adopt `"2.2"`/unsmeared as the CoR-P-faithful form by **registration**, not by equivalence; (b) add **P6** (θ CLI plumbing, §0 F-C) — blocking; (c) re-scope **P0** to the same 2.2/unsmeared form (≈ 5 CPU-h / 40 min, docket §4.1) instead of the smeared 11 CPU-h; the registered smeared b_plus node → REPORTED-ONLY. Optional P1′ (0.33 CPU-h) attributes the −12 % to θ vs smear. |
| 2 | B2: park with the bound | **AGREE** | p = 0.0358 ≥ 0.01 (`cmem_a1_result.json`); A2 not triggered ⇒ k_sky = 1.5 invariant for C3 (L3). |
| 3 | B3: "3.2 warranted" — build the flag + C2 | **DEVIATE: strike C2; accept the STOP** | §F (generator provenance at `03cfe80`, 1514 dark / 76 in-catalogue rows, `seed61000/prepared_cramer_rao_bounds.csv` md5 `9a1f2a14…`) refutes the premise at zero compute; the builder's refusal (`B3_2_POP_FLAG_RECORD.md`) is correct under the approval-scope rule; C2's only residual yield (§6.1 algebra check + Δ_D) is derivable analytically (§6.0) and not worth 45–69 CPU-h. Keep the two deliverables (§13 item 3); assign the B3.1 superseding note; re-cut L1/L4; G7 row 16 re-grade goes to the verifier. |
| 4 | B4: KW-Q1 as registered, behind P1 | **AGREE; P1 is no longer the gate** | driver FT config now built (`hier_s0_driver.py` sha1 `9f831b9f…`, 567+ lines vs `dd63fe0c`) — re-pin L2; the KW-Q1 statistic is form-invariant (`D̃^φ`/`α_G^φ` cancel in `s_imp`, §3 item 9) so the `"2.2"`/unsmeared 8.4 CPU-h form is justified by algebra, not by P1 (`CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3); ENG on `L_cat_no_bh` is non-vacuous (mirror 105/105 moved, §0); three-agent independence stands. |
| 5 | B5: C3 at k = 3, H4, iiib | **AGREE** | registration complete; four minor gaps (§1.4) closable by one appended note; cost 44–137 CPU-h (`COMPUTE_LEDGER.md:45`). |
| 6 | B6: closed at depth 1 | **AGREE** | `1f003da6` landed; bit-identical while `SIGMA_V_PEC_KM_S = 0.0`; L8 satisfied. |
| 7 | B7: C4 = PROD-CF-2D, H4, conditions (a)–(c) | **AGREE** | (a) falsifier (i) PASS (`B7_2_FALSIFIER_I_RECORD.md`: 2.6e-16 / 1.50 / 0.60); (b) STEP-2 smoke registered (§13.2); (c) (ii) excluded (row #220). Commit the test file before the wave-2 commit. |
| 8 | B8: build the harness locally | **AGREE on design; correct the cost** | B8.2 §6: 130–475 CPU-h local (not ≈ 6); S1–S3 can overlap wave 2 (no cluster); S4 registration → verifier; SHARED-FILTER referent must be re-pointed (§3 item 8). |
| 9 | wave-2 batch: C0–C4, 16 tasks, 224–447 CPU-h | **DEVIATE: C0 + C1 + C3 + C4, 13 tasks, 179–357 CPU-h** (+120–173 conditional on a C0 FAIL) | §3 item 7. Launch order: wave-2 commit → C0 + C3 + C4 in one set (C3/C4 baseline reuse is decided by C0's result; their arm-T tasks do not wait) → C1 only after PA-HIER-31 + P6 + P0. |
| 10 | registrations-first list §4.2 items 1–6 | **AGREE, with the gaps of §1 and §5** | item 1 (PA-HIER-31) has a skeleton (§2); item 2 (B3.2) complete but the arm is struck; item 3 (C3) complete − 4 minor; item 4 (C4) complete − 2 minor; item 5 (B8.2) complete as design; item 6 (ledger archive cells) NOT done for any arm. |

---

## 5. GAP list (concrete, ordered by blocking power)

1. **[BLOCKING, all arms] No wave-2 commit exists.** Commit the dirty tree (gate-ledger row, six appended records, `hier_s0_driver.py`, the untracked falsifier test) → that hash is the A22 stamp for C0/C1/C3/C4; append one launch-stamp note to each registration.
2. **[BLOCKING, C0] No C0 registration.** Write the ≤ 1-page note of §1.1 (17-column gate, CLI verbatim + explicit B5.1 defaults, fallback + its cost, what it certifies, verifier line).
3. **[BLOCKING, C1] PA-HIER-31 unauthored** — fill §2; decisions needed from the orchestrator: b-node (±0.033 recommended), θ form (2.2/unsmeared recommended, F-A), P0 re-scope.
4. **[BLOCKING, C1] θ not on the production CLI (F-C)** — add P6 (plumbing commit or production driver) before any S0-B sbatch.
5. **[BLOCKING, C2] No instrument; premise refuted** — strike C2 from wave 2 (agree with §13); re-cut L1/L4 by appended note; append the B3.1 §3 superseding note; drop the stale C2 ledger row (append a "struck" row).
6. **[PROCEDURAL, C0/C1/C3/C4] `COMPUTE_LEDGER.md` archive cells read "pending"** — F4 forbids launch; set "yes" per arm in the launch summary; add the C0-FAIL fallback row (+120–173 CPU-h) and the B8.2 local row (130–475 CPU-h); strike C1's smeared band.
7. **[MINOR, C3] Four appended-note items:** (i) H4 adjudicates on Δmean_h,pred, not ΔMAP; (ii) reconcile R1 ±2 pp vs §8 ±3 SE; (iii) class definition at h = 0.730 vs all-41-node; (iv) one verifier-scope line.
8. **[MINOR, C4] Two appended-note items:** (i) attribution provisional on falsifier (ii) (row #220); (ii) walltime resubmit rule for the STEP-2 overhead pin.
9. **[RECORD, B1] Two chair findings to append (not edit) on `B1_1_HIER_RECORD.md` and the docket:** F-A (site 2.3 reaches the no-BH channel through α_G^φ/D̃^φ; "finding 4" and docket §0(c)/§6 item 2(a) are REFUTED-IN-PART) and F-B (batch-order hypothesis for the 5.7e-4 residual REFUTED; code-delta hypothesis remains).
10. **[RECORD, L2] driver sha1 stale** — re-pin `hier_s0_driver.py` at `9f831b9f7d6b8fed820d547bbe8cd64ff00873e3` (or the wave-2 commit's blob) in the L2 line and the KW-Q1 run record.
11. **[MINOR, B4] KW-Q1 run form** — record in the KW-Q1 run record that `s_imp` is invariant to the `"all"`/smeared vs `"2.2"`/unsmeared choice (§3 item 9) and that the cheaper form was run; no band or statistic changes.
12. **[MINOR, B8.2] SHARED-FILTER falsifier referent** points at the struck C2 arm — re-point at S4.

---

## 6. Numbers with provenance (A11) — everything quoted above that is not a code line

| value | source | date |
|---|---|---|
| HEAD readout commit `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`, 2026-08-27T19:40:20; CoR-P `cli_args` incl. `smear_global_selection: False`, `catalogue_global_selection: phi`, `selection_in_completion_numerator: fused`, `catalogue_numerator_survival_2d: off` | `headreadout_20260827/iiib/run_metadata_21.json` | read 2026-08-29 |
| estimator commits since `d04d9dc9`: `d40fe5c8`, `1f003da6`, `0b308828`, `901653a1`; diff 434+/18− over 5 files | `git log/diff d04d9dc9..dd63fe0c -- darksiren_emri/` | 2026-08-29 |
| P1 read: 9 shared events; `L_cat_no_bh` exact; `alpha_G_phi` 5.8688310e7 → 5.1635200e7; `D_tilde_phi` 9.470921e8 → 9.40039e8; `w_G` 0.06196684 → 0.05492879; `combined_no_bh` max_rel 7.447e-3 | `hier_s0_work/b1_2_smoke/p1_2p2_off/…/node_b_plus_sites2.2_nosmear/…/event_likelihoods.csv` vs `hier_s0_registered_run/s0a_seed900101/node_b_plus/…/event_likelihoods.csv` (pandas merge on `event_idx`) | 2026-08-29 |
| truth 9-vs-106 bit-identity (17 columns) | same dirs, `node_truth*` | 2026-08-29 |
| registered b_plus vs truth (106 events): active 105/105 moved on `L_cat_no_bh`; mean Δln`combined_no_bh` −0.1118 = −0.1193 (kernel) + 0.0075 (global); half-secant score_b proxy mean −5.59, SD 16.88, SEM 1.65; 9-event 2.2/unsmeared proxy mean −5.15, SD 13.97; C-C ratio 1.00750 = 1/0.992553 | same CSVs | 2026-08-29 |
| production classes: 1588 scored; C-C 606; C-A ∪ C-B 982 (0.6184); in-catalogue injected 76 (1514 dark of 1590 CRB rows) | `b3_pop_prediction.json:venues.iiib.{n_dark,n_matched,n_events_scored_csv}`; B3.2 §F item 3 | 2026-08-29 |
| B3.1 bins/edges, measured & predicted per bin, bins 2–5 −0.6123 / −0.6031 (n = 484) | `b3_pop_prediction.json:venues.iiib.{bins,dark_ensemble_bins2to5_only_robustness}`, `registered_bin_edges` | 2026-08-29 |
| B4 q1 edge 0.3575; ft q1 mean `s_imp` −0.798 ± 0.041, share 0.9165; fc share 0.8624 | `b4_imp_stage1_forecast.json:covariates.{ft,fc}.z_true` | 2026-08-29 |
| production O2: full mean_h 0.6077 / MAP 0.60; pure 0.8396 / MAP 0.86; `frac_active_073` 0.6184 | `b4_imp_stage1_production_o2.json:iiib` | 2026-08-29 |
| B5: pass fractions 0.95768 / 0.69509; growth median 0.94855, p95 1.49839, max 10.0, 16 zero→nonzero, 24 zero-under-both; retention 0.95666 → 0.78903 (2163 → 1784 of 2261); jackknife iii mean 0.78977 SD 0.04553 (24 arms) | `b5_window_count.json`, `b5_window_count_arm_jackknife.json:summary_across_arms` | 2026-08-29 |
| pull read: `|pull_def1| ≤ 3` = 0.7877 vs window 0.7890; CV median 1.0182 | `b5_pull_read.json` via `B5_2_PULL_READ_20260829.md` §3 | 2026-08-29 |
| T_mat 0.008 = max(0.005, 0.0239/3); I_HEAD 2965 (σ_h 0.018366) | `MEASUREMENT_HEAD_READOUT_20260827.md:268-285`; proposal §6.2 | 2026-08-27/29 |
| falsifier (i): 2.60e-16 / 1.30e-16 (twin), 1.500 / 5.667 (coded), 0.600 (probe); 52 passed | `B7_2_FALSIFIER_I_RECORD.md` §2 | 2026-08-29 |
| costs: C0 15–23; C1 60–92 (81–113 smeared, withdrawn); C2 45–69; C3 44–137; C4 60–105 (74.7–101.4 incl. C0; ceiling 105/132); anchors 14.93–22.9 CPU-h per h-point | `COMPUTE_LEDGER.md` wave-2 table; docket §4.3; proposal §6.2 | 2026-08-29 |
| B8.2 cost 130–475 CPU-h local; n_U = 100 / 25; PIT–KS band 0.134 | `B8_2_HARNESS_DESIGN_20260829.md` §4.1, §6 | 2026-08-29 |
| PROD-A0 ≤ 8.5e-15 over 12 columns | `BIAS_HISTORY_LEDGER.md:2957` (row #201) | 2026-08-25 |
| b_max 0.0661 (2 × median 0.033038) | PA-HIER-29 | 2026-08-28 |
| S0-A anchors: truth 64.73 s; smeared b_plus 1190.93 s (18.6×); setup ≈ 458 s | `B1_1_HIER_RECORD.md` §2.1, §2.3 | 2026-08-29 |
| driver sha1 now `9f831b9f7d6b8fed820d547bbe8cd64ff00873e3` (L2 cites `5313c319…`) | `sha1sum`; `git diff --stat dd63fe0c` 567+/42− | 2026-08-29 |
| register item 5 [WPOP-TUNING] ≤ +0.0004 at 10 % | `EXONERATION_REGISTER_20260827.md:382-388` | 2026-08-27 |

**Standing-rule-5 note for this check:** no new mechanism is opened here; the S0-B mechanism was
grepped in both layers by the HIER prereg §1.6 and re-run by the B1.1 runner (no match, record §0);
the C3/C4/C2 checks inherit their registrations' own greps (WIN-K3 header; proposal §10; B3.2 §10).

*Chair: inherit-tier subagent, scoped package, 2026-08-29 evening. Files touched by this node: this
file only (new). No git operations; no source edits.*
