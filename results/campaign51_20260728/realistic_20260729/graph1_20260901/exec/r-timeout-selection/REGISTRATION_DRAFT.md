# r-timeout-selection — REGISTRATION DRAFT: does the waveform-timeout truncation reach H0 through p_det (Q1) or through the inference population (Q2)?

Date: 2026-09-04. Node: r-timeout-selection (batch 2; Graph 2 candidates `q-timeout-selection-pdet`,
`q-timeout-population-mismatch`, named in row #355(2); G7 systematics row 8). **DRAFT — PROPOSED
THROUGHOUT, nothing frozen until the author rules.** Author of record for every scientific decision:
Jasper Seehofer. Bands + launch return as fresh RULE **d-timeout-selection-register**. max_revisions 2.
Cost cap: Q1+Q2 ≤ 3 CPU-h local, **zero cluster** (ORCHESTRATOR-DERIVED); §8 lists the NOT-covered
items. Research-cycle stages 0–2; stage-1 forecast in `INFORMATION_FORECAST.md`; the code trace in
`MECHANISM_NOTE.md`. Append-only after commit. The registration author has NOT computed any
registered statistic (§10 leak inventory).

## 0. Claim intake (stage 0)

| hop | what the record says | tag |
|---|---|---|
| `docs/gates/G7_systematics_budget.md` row 8 | "30 s inj / 90 s sim asymmetry; 0.6–1.25 % per stage; bounded sub-% on H₀"; CAMPAIGN: bin by (M, e₀, p₀) | [DOC] — the 30/90 premise is STALE (both 90 s since 2026-07-03, `main.py:1094-1099`) |
| `docs/gates/G9_timeout_scan.md` §1, §7 | `D_i = 1[ρ ≥ ρ_thr]·1[τ ≤ τ_max]`; the τ factor "is not represented in p_det … unless the injection campaign truncates the same events at the same budget" | [DOC] |
| row #342(4), `exec/rd-timeout-bin-seed3000/` | M-axis rate gradient 12.2σ on the seed3000 partial | [DOC], verdict-free |
| row #355(2), `exec/rd-timeout-bin-seed61000/READ_RECORD.md` + chair note | 820/2412 = 0.340; M rates 1.00/0.971/0.144/0.210/1.00; p0 rates 0.232/…/1.00 (p0 > 20 100 %, N = 483); e0 flat; 80.4 % of kept in M-bin 2; "inference population truncated to M ∈ [~2e5, ~3.3e6], p0 ≲ 20" | [DOC], verdict-free; **framing corrected in `MECHANISM_NOTE.md` §3** |
| `MECHANISM_NOTE.md` §1–§3 | timeouts ABSENT from the pool (`main.py:1293-1302` → no `results.append`) and from the kept set (`main.py:763-771`); per-draw SNR-stage rate **822/89,456 = 0.92 %**; the "34 %" is conditional on {kept ∪ timeout}; **p0 axis not evaluable** — the D1 gate (`parameter_space.py:110-111` + `parameter_estimation.py:271-276`) silently removes 4,071/5,921 SNR-passers (no params), so no non-timeout draw with p0 ∉ [10,16] can enter the denominator; kept p0 = [10.0025, 15.987] | [LOCAL] |
| D1 record: `CLAIM_D1_P0WINDOW_20260805.md`, `PREREGISTRATION_D1_SAND_REWEIGHT.md`, ledger row #94 + AUTHOR RULING 2026-08-05 (ledger line 272), row #159 | p0-window = the mass band-pass, 69.3 % of SNR-passers removed, **MIXED — BOUNDED NULL** (does not own the 2D core via the tilt route); bounded null ACCEPTED; transfer-close APPROVED; binding constraint RUNBOOK-7 §1.2b (catalogue stays band-passed; p0-bounds retirement simulation-side, future campaigns only) | [DOC] — **binding; not re-opened here** |

**Claims registered (conjectures).**
- **c-Q1** `q-timeout-selection-pdet`: "the p_det objects, built from completed injection draws only, mis-specify
  the selection of the completed-only event population by ≥ T_mat on mean_h." Counter: c-Q1-shared — the pool
  and the simulation share the truncation (same 90 s budget, generator, draw measure) so p_det is the correct
  selection function of the completed population and the residual (host-mass prior × completed-conditional
  p_det, `MECHANISM_NOTE.md` §4) is < T_mat. `Refute by:` S1.2 + S1.3 on banked tables (zero cluster).
- **c-Q2** `q-timeout-population-mismatch`: "the M-support truncation of the kept population (timeouts, sim-side)
  changes the H0 information or the offset by ≥ T_mat relative to the population the injected prior would have
  delivered." Counter: c-Q2-snr — the kept M composition is set by the SNR threshold, not by timeouts, and the
  per-event influence on mean_h is M-flat. `Refute by:` S2.2/S2.3 on the banked influence vector + CRB CSV.
- **p0 axis: OUT OF SCOPE by construction** — owned by the D1 record (bounded null, ratified). This arm reports
  p0 only as a REPORTED-ONLY covariate and files the §0 correction of the seed61000 read's p0 row.

**Exoneration check (both layers, MECHANISM-grepped; memory `rule1-exoneration-check-insufficient`):**
`CLAIM_2D_BIAS_20260730.md:721-757` and `BIAS_HISTORY_LEDGER.md:127-171` read for "p_det built on a
truncated/selected pool", "timeout", "band-pass/window selection". Hits: (a) ⚠4 "hard support truncation /
hard clamp in production — misspecified under observed-z membership (#63)": an ESTIMATOR-side z-clamp, not a
generator-side waveform truncation — different mechanism; (b) ⚠6 "p_det inside the numerator alone — refuted
(#66)": a placement question, not a pool-composition question; (c) ⚠12 "p_det anchor/first-bin asymptote —
wrong layer (#17)": grid layer, not population; (d) "p_det estimator choice · p_det inside/outside" on the
claim list: same; (e) D1 rows #94/#159 — the p0-window is exonerated as the 2D-core OWNER via the tilt route;
this arm asks a different question (information + composition; the M axis of a different filter). **Not
exonerated; D1's ratified bound is a hard scope fence (no p0 re-litigation).** R0 sweep: G9 note already
cites Mandel–Farr–Gair 2019 ("selection function must match the event-inclusion criterion") — the relevant
literature warning; no new row. Single-agent check → the design-gate verifier re-does it as a decisive claim.

## 1. Populations and data of record (pins — STOP on mismatch)

| object | path (repo-relative) | pin |
|---|---|---|
| production CRB (M, p0, e0, SNR, Fisher cov, `generation_time`) | `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv` | md5 `9a1f2a14384a9281c97ca3be312ddaab`; 1590 rows; scored set = {0..1589} − {1203, 1356} |
| pool of record (a/b/c strata; the p_det denominator) | `…/gate_b_20260730/injection_pool_mix200k_20260728/injection_h_0p73_task_*.csv` | 707 files, 200,100 rows; per-file md5 list md5 `75f4030d5d3b0405fd948049bef5767e` (`POOL_MANIFEST.md5` beside this draft, 707 lines); strata a/b/c = 99,014/50,947/50,139; `code_rev` ∈ {`f6449051` (194,100), `a9f29e82` (6,000, p0/t_plunge NaN)} |
| seed61000 simulate logs (timeouts with params; skip tallies) | `…/seed61000/cluster_logs_fetch_20260904/logs/simulate_6088772_*.{err,out}` | manifest `…/cluster_logs_fetch_20260904_MANIFEST.md5` md5 `ebf09fc4ab66b55e4eb592731ee46ae6`; 100/100 tasks; Σ Y = 89,456 iterations |
| timeout-read bin edges (M, e0, p0; seed61000-native) | `exec/rd-timeout-bin-seed61000/design_gate_bin_edges.json` | md5 `e24b07fe3948559b02d8dd4dbe8df8b3`; M edges [1.147e4, 4.735e4, 1.955e5, 8.074e5, 3.334e6, 1.377e7] |
| **g-c0-baseline** iiib re-baseline CSV (frozen T0 anchor) | `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` | md5 `8e6a2c18dc5838dd1d52641589243672`; 65,108 rows = 41 h × 1588; `den_log_term` is one value per h (verified: 1 unique per h-node) |
| joint_r1 replicate CSV | `…/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv` | md5 `745954a0fdee5f10878fb5e622a06144` |
| per-event influence (frozen T0) | `exec/r-offset-subset/influence_iiib.csv` (`event_idx, influence_2D, influence_1D, rank`) | md5 `d20a01734cc825625f14ba7ec82c67ae`; 1588 rows |
| convention / grid | `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (`_moments`, `_physics_floor_apply`, `w = np.gradient(h_grid)`); `validation/correspondence_1d.py:353` `H_GRID_41` | the frozen T0 convention of rows #302/#342 |
| code of record | HEAD `79c44608` (working tree, read-only); seed61000 simulate commit `03cfe800` (run_metadata_47.json) | line numbers in `MECHANISM_NOTE.md` are HEAD's; the builder re-quotes at `03cfe800` (`git show`) |

Anchors (row #302/#342 JSON): iiib 2D `mean_h 0.6658540600`, `σ_h 0.018474739`; 1D `0.6669870586`; joint_r1 2D
`0.6671274830`. `H_TRUE 0.73` (`constants.py:25`); `SNR_THRESHOLD 20` (`:55`); `T = 4.5 yr`, `dt = 10 s` (CRB CSV).

## 2. Definitions

**M bins** b ∈ {0..4}: the pinned seed61000-native edges (§1), on DETECTOR-frame M (`M` column of both the
CRB CSV and the pool — both store M_z; `simulation_detection_probability.py:492-497`). p0/e0 bins: the pinned
quintile edges, REPORTED-ONLY.

**Completed-draw denominator per bin (Q1).** The pool a-stratum (population measure; `stratum == "a"`,
99,014 rows, p0-NaN rows kept — M is never NaN) is the only local record of completed draws by M_z under the
simulation's own draw measure. Scale `s = N_sim_completed / N_pool_a` with `N_sim_completed = 78,841 + 5,921 =
84,762` (SNR-fail + SNR-pass log counts, both stages of the same alarm). Sim-side timeouts per bin `N_to(b)`
from the 822 logged param dicts (`M` key = M_z, the value FEW saw).
`P_complete^sim(b) = s·N_a(b) / (s·N_a(b) + N_to(b))`. Pool-side `P_complete^pool(b)` needs the pool build's
own timeout tally (`main.py:1349-1352` line, cluster log, NOT local → §8 item A). Ratio
`ρ(b) = P_complete^sim(b) / P_complete^pool(b)`; if item A is not granted, `ρ` is REPORTED as the sim-side
`P_complete^sim` alone with `P^pool ≡ 1` and the disposition is capped at INTERMEDIATE (§5).

**Re-weighted T0 posterior.** For per-event log-likelihood modifications `δ_e(h)` (Q1) or per-event weights
`w_e` (Q2), `ln post'(h) = Σ_e [w_e (ln L_e(h) + δ_e(h))]`, `L_e = combined_with_bh` (2D primary) or
`combined_no_bh` (1D replicate), physics floor and gradient-trapezoid weights exactly as `tier0` (`_moments`);
`Δmean_h = mean_h' − mean_h(anchor)`; `σ'_h/σ_h` alongside. `Σ_e w_e = 1588` always (`g-closure`).

**Per-event information (Q2).** From the CRB CSV: `σ_lnDL = sqrt(delta_luminosity_distance_delta_luminosity_distance)/luminosity_distance`;
sky area `Ω = 2π |sin qS| sqrt(C_qSqS C_φSφS − C_qSφS²)` sr (the qS/phiS block, `_cov_frame` as stored);
`SNR`. Directional influence `d_e = sign(0.73 − mean_h)·(−influence_2D_e)` from the pinned vector (positive =
removing e moves mean_h toward truth), identical to r-offset-subset §2.

## 3. Design — three disjoint agents, formula gate before any real-mode run

Phase A (sonnet, mechanical): parse the 822 timeout dicts + skip tallies → `timeouts_seed61000.csv`
(re-derived, must match the read's `rate_table_M.csv` `n_timeout` column EXACTLY — `g-byteid`); build the
pool a-stratum M_z histogram; write `pcomplete_by_bin.csv` with the `g-closure` sum. Phase B (sonnet):
S1.3 instrument = a standalone script calling library functions ONLY (`SimulationDetectionProbability`,
`precompute_completion_denominator`, `completion_mass_factor_g`/`_g_sel`) with a multiplicative
`P_complete(M_z)` on the p_det values inside the mass quadrature — no edit under `darksiren_emri/`, tagged
`instrumentation`; its own `DESIGN_GATE_formula.md` (verifier, precedent `exec/r-offset-subset/`) must
show, on a synthetic 3-event fixture, that `combined_with_bh` re-assembles from `g_frac`, `B_num`, `L_cat_with_bh`,
`w_G` columns to 1e-9 BEFORE the real CSV is touched (`g-formula`). Phase C (sonnet): Q2 reads (pure pandas).
Adjudication of the three-valued rows: top-tier, verdict-free, returns to the author.

## 4. Registered statistics

**Q1 — `q-timeout-selection-pdet`**
- **S1.1 (existence, banded):** pool timeout tally by M bin from the pool build log (needs §8 item A). Read:
  `P_complete^pool(b)`; also the pool-side per-draw rate. If item A is refused: S1.1 = NOT-EVALUABLE, disclosed.
- **S1.2 (sharing):** `ρ(b)` for every bin with `s·N_a(b) + N_to(b) ≥ 100`; statistic `R = max_b |ln ρ(b)|`
  with Garwood 95 % on `N_to(b)` propagated. Secondary: `P_complete^sim(b)` itself (expected: ≈ 0.98–1.0 in bins
  2–3; the low-M bins are the question — forecast §2 of `INFORMATION_FORECAST.md`).
- **S1.3 (materiality, PRIMARY):** `Δmean_h^{Q1}` (iiib 2D) from `δ_e(h) = ln g'_frac,e(h) − ln g_frac,e(h)`
  where `g'` uses `p_det × P_complete^sim(M(1+z))` inside the completion-leg mass quadrature and the catalogue-leg
  `mz` expectation, at every h-node and event; plus `δ^den(h) = −[ln D'(h) − ln D(h)]` from the M_z-weighted
  pool (weights `ρ(M_z)`; ≡ 0 if S1.1 is NOT-EVALUABLE — then the denominator leg is REPORTED as the
  `P^sim`-weighted bound). Replicates: iiib 1D (denominator leg only), joint_r1 2D. Reported alongside:
  `σ'_h/σ_h`, MAP', per-bin decomposition of Σ_e δ_e(0.73) by M bin ([A2] paired read).
- **S1.4 (engagement, [A13]):** ≥ 10 % of events must have `|δ_e(0.73)| ≥ 1e-6`; else INSTRUMENT (no null read).

**Q2 — `q-timeout-population-mismatch`**
- **S2.1 (information map, REPORTED-ONLY):** per M bin (and p0/e0 bins): n, median and IQR of `σ_lnDL`, `Ω`,
  `SNR`, `generation_time`; Spearman `ρ_S(log10 M, ln σ_lnDL)` with a 10,000-permutation p.
- **S2.2 (influence vs M, banded):** Spearman `ρ_S(log10 M, d_e)` and `ρ_S(log10 M, |d_e|)`, permutation p
  (10,000 draws, seed `20260904`); the top-k = 82 subset's M-bin composition vs the bulk (Fisher exact per bin,
  Holm over 5). Replicates: 1D (k = 94), joint_r1 2D (k = 72).
- **S2.3 (composition counterfactual, PRIMARY):** `w_b = share_pool,det(b) / share_kept(b)` over bins with
  `n_kept(b) ≥ 10`, where `share_pool,det(b)` is the M-bin share of pool a-rows with `SNR ≥ 20` (the injected
  prior's DETECTED population, i.e. "what the population would have been" without the sim-side filters) and
  `share_kept(b)` the kept share; `w_e = w_{b(e)}` renormalised to Σ = 1588. Read `Δmean_h^{Q2}`, `σ'_h/σ_h`
  (iiib 2D primary; 1D + joint_r1 replicates). Same-size null: 1000 draws of `w_e` permuted over events (seed
  `20260904`) → `Δ_null` SD. Decomposition REPORTED-ONLY: `share_to(b)` of the 822 timeouts, and the D1-gate
  share (unmeasurable per bin — disclosed) so the reader sees which sim-side filter the composition gap
  belongs to. NOTE: `w_b` re-weights only bins with kept support; bins 0 and 4 (0 kept) cannot be
  re-created — the counterfactual is a bound over the supported range (structural blindness, §6).
- **S2.4 (hypothesis line, REPORTED-ONLY):** timeouts' `(log10 M, p0, mu/M)` scatter vs the kept set; the
  M–p0 coupling through the plunge-window map (`MECHANISM_NOTE.md` §5).

## 5. Bands (ORCHESTRATOR-DERIVED) and three-valued dispositions — every row returns as a fresh RULE

`T_mat = 0.008` (the materiality band of rows #342(5)/#351, = 0.43 σ_h). `T_null = max(0.002, 2·SD(Δ_null))`.

| row | statistic | MATERIAL | IMMATERIAL | INTERMEDIATE |
|---|---|---|---|---|
| Q1-S1.2 | `R = max_b |ln ρ(b)|` | `R ≥ 0.20` in any bin with ≥ 100 draws → NOT-SHARED | `R ≤ 0.05` everywhere → SHARED-FILTER | else |
| **Q1-S1.3** | `|Δmean_h^{Q1}|` iiib 2D | `≥ T_mat` → **P_DET-MISSPECIFIED-MATERIAL** | `≤ T_null` AND replicates agree in sign or are ≤ T_null → **P_DET-MISSPECIFIED-IMMATERIAL** | else → **P_DET-MISSPECIFIED-INTERMEDIATE** |
| Q2-S2.2 | `ρ_S(log10 M, d_e)` | `p_perm < 0.01` AND top-82 vs bulk Fisher Holm-p < 0.05 in ≥ 1 bin → M-STRUCTURED | `p_perm ≥ 0.10` → M-FLAT | else |
| **Q2-S2.3** | `|Δmean_h^{Q2}|` iiib 2D; `σ'_h/σ_h` | `≥ T_mat` OR `σ'_h/σ_h ∉ [0.80, 1.25]` → **POPULATION-MISMATCH-MATERIAL** | `≤ T_null` AND ratio ∈ [0.95, 1.05] → **POPULATION-MISMATCH-IMMATERIAL** | else → **POPULATION-MISMATCH-INTERMEDIATE** |

Caps: if S1.1 is NOT-EVALUABLE, Q1's best disposition is INTERMEDIATE ("consistent with shared filter, pool
side unverified") — MATERIAL is still reachable (a material `Δ` from the sim-side bound alone is material).
If S1.4 fails, Q1 = INSTRUMENT, no disposition. Rail rule: any MAP' at 0.60/0.86 turns that Δ into a BOUND.
Mandatory line in every disposition: "p0 axis not evaluated (D1 record, ratified bound; read's p0 row
corrected per `MECHANISM_NOTE.md` §3)". **Fresh RULE on each of the four rows; none pre-decided.**

## 6. Gates

- **G-1 pins:** every md5/count in §1; STOP on mismatch (CLAUDE.md dataset-pinning rule).
- **g-byteid:** phase A's per-bin `n_timeout` = the read's `rate_table_M.csv` column EXACTLY (206/302/216/81/15);
  its `n_kept` = 0/9/1279/304/0; the T0 anchor re-computed from the iiib CSV to |Δ| ≤ 1e-9 (`0.6658540600`);
  the pinned influence vector reproduced to 1e-12 for the top-10 of row #342 JSON. Any miss = INSTRUMENT.
- **g-population:** sim logs 100/100 tasks, Σ Y = 89,456 (±0), 822 timeout dicts (820 + 2), 4,071 D1-gate lines;
  pool 200,100 rows / 99,014 a-rows / 6,000 p0-NaN rows (kept for M, excluded for p0 reads, disclosed); 1588
  scored events per h × 41 nodes, no G-EXT nodes; kept p0 ∈ [10.0025, 15.987] re-asserted (if a kept row lies
  outside, the D1 reading of `MECHANISM_NOTE.md` §3 is wrong → STOP).
- **g-closure (re-weightings must sum):** (i) `Σ_b [s·N_a(b) + N_to(b)] = 84,762 + 822 = 85,584` exactly by
  construction, and the log-derived iteration count `89,456 − 3,488 (ZeroDiv) − 85,584 = 384` residual must be
  accounted (pre-screen + CRB-other + walltime-cut partial iterations) — residual > 1,000 = STOP; (ii) Q2
  `Σ_e w_e = 1588` to 1e-9; (iii) Q1 `δ^den(h)` applied identically to all 1588 events (one scalar per h).
- **g-precision:** float64 log-sums; frozen T0 exactly; 0 physics-floor exclusions expected (STOP if ≠ 0);
  `combined_*` columns only (full precision), never the 7-s.f. columns for likelihood assembly.
- **g-formula (Q1 S1.3):** the synthetic re-assembly check of §3 committed with its hand arithmetic BEFORE the
  real run; the `P_complete` factor enters ONLY inside `p_det` calls within mass quadratures (never the 1D
  no-BH numerator survival, never the catalogue candidate survival at fixed observed M) — the verifier lists
  every call site the script reaches.
- **g-hardware:** the seed61000 node list (`.out` "Node:" lines, `uc2n561…579`) mapped to GPU type by one
  cluster read (`scontrol show node`, §8 item B; chair-only). Reported alongside S1.2; if H100 share > 0, the
  sharing claim is conditional on it, by name.
- **g-scope:** no statistic may be computed on p0 bins other than S2.1/S2.4 REPORTED-ONLY rows; no re-scoring
  of the catalogue against band-blind objects (RUNBOOK-7 §1.2b).

**Invariants ([A10]):** frozen T0 convention (audited rows #302/#342, 2026-09-03) · `H_GRID_41` · `h_true 0.73` ·
90 s alarm in both loops (audited today, `main.py:619/792/1099`) · `T = 4.5 yr`, `dt = 10 s` (CRB CSV; pool
assumed equal — `NEVER` audited on the pool side: the pool CSV carries no T/dt column; conditional by name) ·
pool M = M_z convention (`simulation_detection_probability.py:492-497`) · D1 window [10, 16] untouched · the
pinned bin edges · seed `20260904`. **Structural blindness:** (1) the logs carry no parameters for SNR-failed
draws or D1-gate drops, so every sim-side per-bin rate is reconstructed through the pool a-stratum under
the identical-draw-measure assumption — a draw-measure drift between the pool build (`f6449051`) and the
simulate commit (`03cfe800`) is invisible to this design except through `g-closure`; (2) Q2 cannot
re-create events in bins with zero kept support — MATERIAL there can only be bounded from bins 1–3; (3) a
timeout that correlates with SNR at fixed (d_L, M_z) (fast-completing draws being systematically louder or
quieter) changes `S(d_L | M_z)` itself; only the §8 rescue run can see it.

## 7. Cost (ORCHESTRATOR-DERIVED, cap ≤ 3 CPU-h, zero cluster)

Log parse 100 files (seconds); pool load 200k rows (≈ 10 s); T0 re-weightings 41 × 1588 (ms); Q1 S1.3: the
`p_det` object build (≈ 1–2 min, as in every evaluate) + `completion_mass_factor_g` at 41 h × 1588 events ×
G-node quadrature (production timing 5–7 min per h-point on a cluster node for the FULL likelihood; the
g-factor alone is a small fraction) — budget ≤ 2 CPU-h; nulls 1000 × 41 × 1588 (seconds). Headroom ≥ 1.5×.

## 8. NOT-covered items for the author (each a [DO] with a cost band; none launched by this registration)

- **A. Pool-side timeout tally** — fetch the `injection_pool_mix200k_20260728` build logs (the
  `"Injection campaign complete: … N timeouts @ 90s"` lines + any params dicts) from the workspace (expires
  2026-09-23). Cost: one rsync, 0 GPU-h. Unlocks S1.1 and the SHARED/NOT-SHARED read; without it Q1 is capped.
- **B. GPU type per seed61000 node** (`scontrol show node uc2n5xx`): 0 GPU-h. Unlocks `g-hardware`.
- **C. Rescue re-run of the 822 logged timed-out parameter sets** at a 600 s budget (SNR only, no CRB), on
  BOTH an A100 and an H100 node: the direct measurement of `P(complete | budget, hardware)` and the
  completion-time distribution — decides the physical hypothesis (`MECHANISM_NOTE.md` §5) and whether a
  longer runtime constant is the fix. Cost band: 822 × [20 s … 600 s] ≈ **5–140 GPU-h** worst case, ×2
  hardware; the G9 timing basis (`inject.sbatch` 30-min A100 tasks; ≈ 2000 tasks per 200k pool,
  `docs/campaign_redesign_51_design.md:219`) puts a same-size pool at ≈ 1000 GPU-h — the rescue is ≤ 14 %.
- **D. A longer timeout as "the fix"** — a runtime constant (`main.py:619/792/1099`), NOT a physics change,
  but adopting it means a NEW pool AND a new simulate campaign under the same budget (≈ 1000 + 50 GPU-h at
  the current rates, more with the rescued slow tail), plus a fresh pin era. It changes the population the
  paper reports; it must not be applied to one side only (G9 §2 D2 is the precedent failure). Author's call,
  after C.
- **E. G7 row 8 text** — the "30 s / 90 s" premise and the "sub-%" magnitude are STALE; append-only note
  proposed after the arm reads out (docs-only [DO]).

## 9. Open questions routed to d-timeout-selection-register (fresh RULE)

1. Ratify `T_mat = 0.008`, `T_null`, the S1.2 0.05/0.20 ln-ratio bands, the S2.3 width band [0.80, 1.25].
2. Grant/deny §8 A and B (zero-GPU cluster reads) before launch — they set Q1's ceiling.
3. Accept the §0/`MECHANISM_NOTE.md` §3 correction of the seed61000 read's p0 row (a chair action on the record:
   "13.6σ" → NOT-EVALUABLE by construction) — [RULE] on the record, not on physics.
4. Confirm the p0-axis scope fence (D1 record owns it) — or re-open D1 explicitly (NOT recommended here).
5. Whether §8 C is wanted before or after this arm reads out (C makes S1.2's sharing question moot on the
   hardware axis).

## 10. Blindness status and leak inventory (binding)

The registration author read code and banked tables to DEFINE the statistics. Pre-reads that touch a
registered object, disclosed: (i) pool a-stratum support fractions (p0 < 10: 21.2 %; p0 > 16: 46.6 %;
M < 1.955e5: 29.8 %; M > 3.334e6: 4.9 %) — inputs to S1.2's denominator, not S1.2 itself; (ii) **one S2.3
input pre-read**: the pool a-rows with SNR ≥ 20 (n = 7,548) have 82.7 % in M-bin 2 vs the kept 80.4 % —
one bin of `share_pool,det`; the registration author did NOT compute `w_b`, any re-weighted posterior, any
Δmean_h, `ρ(b)`, `P_complete`, `δ_e`, or any influence–M correlation; (iii) the log tallies of
`MECHANISM_NOTE.md` §3 (these are `g-population` gate values, quoted there). No statistic of §4 was run.
Chair/author: treat (ii) as a partial pre-read of S2.3 in the same sense as row #344 was for r-offset-subset.

## 11. Design-gate self-check

Stages 0–2 present; both exoneration layers grepped by mechanism; every claim has `Refute by:`; three-valued
rows with fresh-RULE tags; invariants + structural blindness listed; pins with md5; `g-closure` sums stated;
costs zero-cluster with NOT-covered items separated and cost-banded from the G9/inject.sbatch timing
basis; the D1 scope fence is binding; the read's p0 row correction is filed as a record action, not a
physics claim. Line count ≤ 320.

## REVISION 1 (Q2) — 2026-09-04, answering `DESIGN_GATE_Q2_computability.md` F1–F4 (append-only; supersedes the named passages above; no threshold or band touched)

- **F2/F3 — ONE support rule for S2.3, fixed now.** A bin is SUPPORTED iff `n_kept(b) ≥ 10`; on the pinned
  counts `0/9/1279/304/0` the supported set is **bins {2, 3} only** (two bins, not three). Events in any
  UNSUPPORTED bin (bins 0, 4: none; bin 1: the 9 kept events) receive **`w_e = 1` — no re-weighting** — and the
  single renormalisation `w_e ← w_e · 1588 / Σ_e w_e` runs over ALL 1588 events, so `g-closure` (ii) holds by
  construction. Justification (one sentence): re-weighting 9 events from a 302-timeout bin would manufacture
  a counterfactual from a share estimate with ~30 % Poisson error, and folding bin 1 into bin 2 would
  silently change the pinned edges that S2.1/S2.2 and the timeout read share — `w_e = 1` does neither and is
  disclosed as a BOUND. The S2.3 NOTE ("bins 0 and 4 … supported range") and §6 structural blindness (2)
  ("bounded from bins 1–3") are corrected to: **"the counterfactual is a bound over bins 2–3 (1583 of 1588
  events); bins 0, 1, 4 are unsupported and carry `w_e = 1`, disclosed."** `share_pool,det(b)` and
  `share_kept(b)` are both computed over bins 2–3 only (renormalised to sum to 1 over that support) before
  forming `w_b`. No builder choice remains on the PRIMARY statistic.
- **F1 — missing pin added to §1:** `exec/r-offset-subset/influence_joint_r1.csv` (`event_idx, influence_2D,
  influence_1D, rank`), md5 **`38f3f1813a3d460093763dd89019ca8a`** (verified on disk by this revision; the
  gate note's `…8a4` carries a stray trailing character), 1588 rows — the k = 72 joint_r1 replicate input.
  All other Q2 inputs are already pinned: the 1D replicate reads `influence_1D` from the pinned
  `influence_iiib.csv`; the 822 timeout dicts and skip tallies come from the pinned log manifest; bin edges,
  CRB CSV and both `event_likelihoods.csv` are pinned in §1. G-1 now STOPs on this file too.
- **F4 — `|d_e|`'s role:** `ρ_S(log10 M, |d_e|)` is **REPORTED-ONLY** and does not gate the Q2-S2.2
  disposition; the disposition is driven solely by `ρ_S(log10 M, d_e)` (permutation p) AND the top-k
  Fisher/Holm composition test, exactly as the §5 row states.

## REVISION 2 (Q2) — 2026-09-04, answering `DESIGN_GATE_Q2_computability_rev1.md` F5 + AMBER (append-only; no threshold or band touched)

- **F5 — `g-byteid` n_kept anchor RE-PINNED to the pinned inputs.** The former target `0/9/1279/304/0` came from
  `rd-timeout-bin-seed61000/rate_table_M.csv`, which folds the 2 CRB-stage timeout records (M = 576074.30 →
  bin 2; M = 1950892.90 → bin 3) into "kept". Derivation of record, from §1's pins only: histogram of the CRB
  CSV `M` column (md5 `9a1f2a14…`) over `seed61000_M_edges` (md5 `e24b07fe…`), restricted to the 1588 scored
  events (event_idx {0..1589} − {1203, 1356} — the population every Σ w_e spans) → **`n_kept = [0, 9, 1276,
  303, 0]`** (reproduced by this revision; over all 1590 CRB rows it is `[0, 9, 1278, 303, 0]`). `g-byteid`
  now targets `[0, 9, 1276, 303, 0]` EXACTLY; `share_kept(b)` and the §REVISION-1 support rule use the same
  1588-event histogram. `rate_table_M.csv` (md5 `b0d6284c06eb2f185158819d47123de5`) is retained REPORTED-ONLY
  as the source of the read's `n_timeout` column (206/302/216/81/15, still the phase-A byte-id target for
  timeouts); its `n_kept` column is NOT a target — it includes the 2 CRB-stage timeouts, disclosed. Supported
  set unchanged: bins {2, 3} (1276 + 303 = 1579 re-weighted; bins 0/1/4 = 9 events at `w_e = 1`; the
  REVISION-1 figure "1583 of 1588" is corrected to **1579 of 1588**).
- **AMBER — §10 item (ii) restated on one bin set.** Pool a-stratum rows with SNR ≥ 20 (n = 7,548) bin as
  `[76, 1217, 4387, 1852, 16]`: bin 2 alone = **58.1 %**, bins 2+3 = **82.7 %**; kept (scored 1588): bin 2 =
  1276/1588 = **80.4 %**, bins 2+3 = 1579/1588 = **99.4 %**. The earlier sentence compared pool bins 2+3 with
  kept bin 2 — corrected; the leak disclosure stands (still one pre-read of a `share_pool,det` input).

**Erratum for the chair to carry to `rd-timeout-bin-seed61000/READ_RECORD.md`:** its per-bin `n_kept`
(`rate_table_M.csv`, `selection_effect_note.csv`, denominators) includes the 2 CRB-stage timeout records as
"kept"; on the pinned CRB CSV the kept counts are `[0, 9, 1278, 303, 0]` (1590 rows) / `[0, 9, 1276, 303, 0]`
(1588 scored) — rates and gradients unchanged at the quoted precision.

## CHAIR ERRATUM (append-only, 2026-09-04 ~03:35 CEST; closes gate rev2 F6; no threshold touched)
The pinned n_timeout anchor [206,302,216,81,15] is the population of the 820 SNR-stage timeout
records. Phase A parses ALL 822 records but bins them by stage: the SNR-stage 820 form the g-byteid
target; the 2 CRB-stage records (M = 576074.30 → bin 2; M = 1950892.90 → bin 3) are listed
separately as reported-only and never enter n_timeout or the S2.3 decomposition line, which is
restated as "share_to(b) of the 820 SNR-stage timeouts (+2 CRB-stage, reported)". The draft's
max_revisions 2 counted the two pre-launch registration revisions; the revision counter for
post-disposition re-registration (charter §1.13) is untouched (0 consumed). LAUNCH of Q2 is deferred
to the morning docket as [DO] R18 (zero compute; gate expected GREEN on this erratum).

## REVISION 1 (Q1) — 2026-09-04, answering `DESIGN_GATE_Q1_computability.md` F1–F6 (append-only; supersedes the named passages; no threshold or band touched)

**Revision counter.** Pre-launch design-gate revisions (this and the Q2 revisions above) are computability
fixes made BEFORE any registered read; they do NOT consume the charter's post-disposition revision counter —
the header's `max_revisions 2` remains unspent by Q1/Q2 design rounds.

1. **S1.1 source (F1).** The M-binnable pool-side timeout record is the per-draw `TimeoutError` catch warning
   (`main.py:1293-1302` at HEAD; logged as `[main.py:1143 - injection_campaign()] Injection waveform/SNR
   computation timed out (>90s, N total). Skipping event... params={'M': …, 'p0': …}` at the pool-build
   commit) — its `M` key is M_z, binned on the pinned edges. 5,040 raw lines in the fetch = 2,520 unique
   events after removing the `.err`/application-log duplicate (dedupe on timestamp + params string). The
   per-task aggregate line (`main.py:1349-1352`, "… N timeouts @ 90s") is the **g-closure tally only**.
2. **Pool-side population rule (F2).** S1.1's numerator AND denominator use **exactly the 707 task attempts
   whose CSV is in `POOL_MANIFEST.md5`**, matched mechanically by (seed dir, `SLURM` task id) from
   `run_metadata_*.json` ↔ `injection_h_0p73_task_<id>.csv`; the other 363 attempts (326 old-format
   completion lines + 37 crashed, = 1,070 − 707; consistent with the pool's `code_rev` split
   `f6449051`/`a9f29e82`) are EXCLUDED and their counts disclosed. Justification: only those 707 attempts'
   completions are in the denominator, so only their timeouts belong in the numerator. g-closure target:
   Σ of the 707 tasks' aggregate "N timeouts" = the count of deduplicated per-draw lines in the same 707
   logs, EXACTLY (STOP on mismatch). Disclosed figures over all attempts: aggregate Σ over the 1,033
   complete-line tasks = 2,475 vs per-draw lines over all 1,070 attempts = 2,520; the 45 are attributed to
   the 37 crashed attempts (per-draw lines logged, no tally line) — reported, never used.
3. **S1.3 denominator leg (F3) — re-registered.** `precompute_completion_denominator`
   (`bayesian_statistics.py:1170-1324`) has NO mass axis: its integrand (`:1284`) is the M-marginal
   `S(d_L)` × `dVc/(1+z)`; an M-dependent `P_complete` cannot enter it as a quadrature factor (the
   mass-integrated rate is the z-independent constant of `:1292-1297`). `D'(h)` is therefore **dropped from
   the primary**. The with-BH per-host denominator that DOES carry a mass quadrature is
   `_mass_trunc_denominator_inner_m_integral(z, detection_probability, host_phiS, host_qS, host_M, sigma_lnM,
   Z_M, h)` (`:869`; batch twin `:944`), called from `denominator_integrant_with_bh_mass` (`:8048-8058`) —
   a Gauss–Legendre integral in ln M over the host prior, p_det queried at `:901`. **Zero-edit insertion
   point:** its `detection_probability: Any` parameter is duck-typed — the instrument passes a PROXY object
   wrapping the production `SimulationDetectionProbability` whose
   `detection_probability_with_bh_mass_interpolated(d_L, phi, theta, M_z, **kw)` returns
   `p_det × P_complete(M_z)` (bin-step in M_z on the pinned edges) and whose every other attribute/method
   delegates unchanged. The primary `δ_e(h)` is the per-event change of `ln combined_with_bh` re-assembled
   from the proxy-evaluated numerator (item 4) and this per-host denominator. The M-marginal
   `D(h)`/`S(d_L)` **composition leg** is SECONDARY: a rejection-thinned pool copy (row kept with probability
   `ρ(M_z)/max ρ`, seed `20260904`, a derived directory à la `d1_sand/make_sand_pools.py`, tagged
   `instrumentation`) fed to an unmodified `SimulationDetectionProbability` + `precompute_completion_denominator`;
   it is identically zero when S1.2 = SHARED (`ρ ≡ 1`) and is REPORTED alongside.
4. **Numerator leg + call-site table (F4).** Registered insertion mechanisms, zero edits under `darksiren_emri/`:
   (a) `completion_mass_factor_g_sel(z_nodes, d_L_gpc, d_L_fraction, det_M_z, proj_d_L_to_M, sigma_cond_M, *,
   s_query: Callable[[d_L, M_z, z], survival], n_hermite, adaptive)` (`:2276`) — the instrument injects
   `s_query' = lambda d, m, z: s_query(d, m, z) · P_complete(m)`; (b) the proxy object of item 3 for every
   other with-BH site. Rule (supersedes the §6 g-formula clause "never the catalogue candidate survival at
   fixed observed M"): `P_complete(M_z)` multiplies **every with-BH p_det query** — a completed-only
   selection applies to any source of mass M_z, catalogue candidate or completion node alike (MFG 2019: one
   selection function for the whole inclusion criterion) — and **no without-BH query** (M-marginal
   survivals move only through pool composition, item 3 secondary). Registered table for the g-formula gate:

   | site | enclosing function | accessor | inside a mass quadrature? | role | mechanism |
   |---|---|---|---|---|---|
   | `:901` | `_mass_trunc_denominator_inner_m_integral` | with_bh | YES (G-L in ln M, host prior) | per-host denominator | proxy |
   | `:944` | `…_inner_m_integral_batch` | with_bh | YES | batch twin | proxy |
   | `:1284` | `precompute_completion_denominator` | without_bh | NO (z-only) | `D(h)` | composition leg only |
   | `:1440` | `precompute_missing_completion_denominator` | without_bh | NO | missing-completion `D` | composition leg only |
   | `:1741` | `_smeared_global_pdet_expectation` | with_bh | NO (fixed catalogue `M_g`, z nodes) | Σ_glob with-BH | proxy |
   | `:1770` | `_smeared_global_pdet_expectation` | without_bh | NO | Σ_glob no-BH | composition leg only |
   | `:2058` | `precompute_phi_marginal_survival` | with_bh | YES (`M_grid` × z grid → `S̄_φ`) | φ-marginal survival | proxy |
   | `:3029` | `precompute_global_catalog_selection` | with_bh | NO (fixed `M_g`) | Σ_glob,wbh | proxy |
   | `:3066` | `precompute_global_catalog_selection` | without_bh | NO | Σ_φ | composition leg only |
   | `:5567` | `_collect_candidate_dump_rows` | with_bh | NO (scalar, per candidate) | DIAGNOSTIC dump, not likelihood-bearing | proxy (inert) |
   | `:2276` | `completion_mass_factor_g_sel` (`s_query`) | with_bh via Callable | YES (Hermite in `x_M`) | completion-leg `g_sel` numerator | `s_query'` |

   The gate verifier must confirm, by the engagement assertion of S1.4, that the proxy is reached at each
   with-BH site production actually dispatches ([A13]) and that `combined_with_bh` re-assembles to 1e-9.
5. **g-closure(i) arithmetic (F5) + manifest pin (F6).** Corrected: SNR-stage ZeroDivisionError = **3,449**
   (the 3,488 of `MECHANISM_NOTE.md` §3 is SNR + 39 CRB-stage); residual = **89,456 − 3,449 − 85,584 = 423**
   (STOP threshold 1,000 unchanged). New §1 pin: `gate_b_20260730/injection_pool_mix200k_20260728_buildlogs_fetch_20260904/MANIFEST.md5`,
   md5 **`6ae9c1098c1c3325504e4904b2fc4d50`**, 3,510 lines; `md5sum -c` must report 3,509 OK and exactly ONE
   failure — the self-referential `./MANIFEST.md5` row (`d41d8cd9…`, the empty-file hash written before the
   manifest existed), a benign construction artefact; any other failure = STOP. §8 item A is thereby
   DISCHARGED (fetch present); item B (GPU type per node) remains open.
