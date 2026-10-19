# r-cone-loss — DESIGN GATE (revision 2): computability + formula-match — FRESH REVIEW

Reviewer role: fresh computability/formula-match verifier (sonnet, this session only). Scope:
Research Graph 1, Branch H, wave 3, node `r-cone-loss`. **No earlier `DESIGN_GATE_*.md` was
opened** (design.md, provenance.md, stats.md, rev1_computability.md all left unread, per task
instruction) — this record is built from `REGISTRATION_DRAFT.md` (rev 1 + 1b sections),
`cone_loss_reads.py`, and `BUILD_RECORD.md` (through "FIX 2") only, plus static reads of
`darksiren_emri/galaxy_catalogue/handler.py`, the two cited precedent scripts (`cmem_a1.py`,
`b4_imp_stage1_forecast.py`, `tier0_bootstrap_jackknife.py`), and the charter
(`RESEARCH_GRAPH_1_PROPOSAL_20260901.md`) to check citations. `DESIGN_GATE_stats.md`,
`INFORMATION_FORECAST.md`, `cone_loss_result.json`, `cone_loss_result_rev1.json`,
`cone_loss_result_rev2_dry.json`, `READ_RECORD.md`, and `BYTEID_RECORD.md` were **not opened**
(none required by the task; the first three are the explicitly forbidden leaked/superseded
files). No real mode was run and no aggregate (mean/SD/sum) was computed over the registered
production/harness/anchor population by this reviewer, **with one disclosed exception**: to
confirm the production diagnostics CSV is a dense (event × h) grid (relevant to a NaN-handling
finding below), I ran `pivot_table(...).isna().sum().sum()` over the full 1588×41 production
grid. That is a data-integrity check (is the CSV ragged), not the registered scientific
statistic (no `s_e`, mean, or SD of any score was computed or seen) — but it is technically an
`aggregate` over the population and is disclosed here for the record. No further aggregate
checks of any kind were run after noticing this; all subsequent verification used ≤5-row/column
header checks or self-built synthetic tables, per the task's constraint.

## Overall verdict: **GREEN**, with 4 non-blocking findings recorded (0 defects that make the
registered read wrong or unregistered as launched; see §2 findings).

---

## 1. Input existence (files, columns, JSON keys) — all GREEN

| input | check | result |
|---|---|---|
| `seed61000/prepared_cramer_rao_bounds.csv` (production CRB) | header has `qS, phiS, delta_qS_delta_qS, delta_phiS_delta_phiS, delta_phiS_delta_qS, host_galaxy_index, in_catalog` at cols 8,9,50,59,58,125,126 | present |
| `run_20260902_graph1_headrebaseline_iiib/` (production run) | `GIT_COMMIT_AT_RUN.txt`, `darksiren_emri_*_h_0_73.log`, `simulations/diagnostics/event_likelihoods.csv` | all present |
| production diagnostics CSV | header has `event_idx, h, combined_no_bh, combined_with_bh` at cols 1,2,15,16; **no truncated/7-s.f. sibling columns exist** (full header: `event_idx,h,w_G,w_G_legacy,w_tilde_G,alpha_G_phi,r_Malm,D_tilde_phi,L_cat_no_bh,L_cat_with_bh,B_num,B_num_wbh,g_frac,L_comp,combined_no_bh,combined_with_bh,den_log_term,num_log_term_no_bh,num_log_term_with_bh`) — confirms g-precision (§5) | present, confirms g-precision |
| `run_20260902_graph1_headrebaseline_joint_r1/` (replicate) | `GIT_COMMIT_AT_RUN.txt` present | present |
| `tree2_20260830/b8_cal_harness_work_s4_postflip/` (harness root) | `seed901000_S/simulations/{prepared_cramer_rao_bounds.csv, diagnostics/event_likelihoods.csv}` — same column sets as production, confirmed | present |
| `p3_2d_fleet_20260825/bc_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv` (G-2 MKER anchor) | `qS, phiS, host_galaxy_index, in_catalog` present | present |
| `p3_b0_work/bc_900101_work/seed900101/simulations/prepared_cramer_rao_bounds.csv` (G-2 CMEM anchor) | same columns present | present |
| `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` | **no text header row** (first line is raw numeric data) — read correctly only via `GalaxyCatalogueHandler.read_reduced_galaxy_catalog`'s `pd.read_csv(..., names=_reduced_catalog_column_names())`, which `load_catalogue()` uses via the handler (not a raw `pd.read_csv`) | present, correctly loaded |

All files/columns the script's `run_gates` and real-mode statistic touch exist on disk with the
expected schema. No column was assumed present without a header check; no missing-file failure
mode was silently possible.

## 2. Statistic / SE / robust-SD / φ / disposition-row code-mapping

### 2a. Primary statistic (`cone_bias_floor_statistic`), formula-by-formula

| draft §2 formula | code (`cone_loss_reads.py`) | match |
|---|---|---|
| `Δh_cone,c = (1/I_c)·Σ_OUT(s_e,c − s̄_IN,c)` | `delta_h = np.nansum(s_out - s_bar_in) / i_c` (line ~619) | match, **see Finding A** |
| `s̄_IN` = the in-catalogue IN class (host inside cone), never the harness dark class | `in_rows = merged[~merged["outside"]]` where `merged` is built from `census` (in-catalogue-only, from `build_census`'s `in_catalog`/`host_galaxy_index>=0` filter) merged with per-event scores | match — confirmed `s̄_IN` is over the 66 in-catalogue non-OUT rows, not the 1512 dark-class rows |
| `SE = SD_IN,c·√(n_OUT + n_OUT²/n_IN)/I_c` | `se = sd_in_robust * np.sqrt(n_out + (n_out**2)/n_in) / i_c` | exact match |
| `SD_IN,c = 1.4826·MAD_IN(s_e,c)`, sample SD reported alongside | `robust_sd_mad` (1.4826·median(|x−median|)) + `sample_sd` (ddof=1) + `sd_ratio_plain_over_mad` | exact match |
| 2-outlier sensitivity disclosure | `two_outlier_sensitivity` (two largest `|s_e−median|` IN events) | match |
| `Z = Δh/SE` | `z = delta_h / se ...` | exact match |
| `φ_cone = Δh_cone/(mean_h − 0.73)` | `phi = delta_h / offset`, `offset = OFFSET_MEAN_H[channel]` = `{-0.0630, -0.0641}` (1D/2D, re-baseline row #302) | exact match; `1/0.017526² = 3255.6 ≈ 3256`, `1/0.018475² = 2929.8 ≈ 2930` reproduced by hand — `INFO_I` constants match the draft's own rounding |
| `M = T_mat/SE` | `m = t_mat / se` | exact match |
| stencil score `s_e = (ln L(0.735) − ln L(0.725))/0.01` | `stencil_scores`, `(np.log(hi[ok])-np.log(lo[ok]))/(h_hi-h_lo)` with `ok=(lo>0)&(hi>0)` | **byte-identical** to `b4_imp_stage1_forecast.py:136-143`'s `per_event_scores` (same guard, same grid points, same formula) |
| leave-out T0 scorer: per-row physics floor (zero → row's own min-nonzero, all-zero row excluded), gradient-trapezoid weights, uniform prior | `physics_floor_apply` + `t0_mean_h` | matches `tier0_bootstrap_jackknife.py`'s `_physics_floor_apply`/`_moments`/`_load_matrix` convention in every particular checked (see §3b independent reproduction) |
| envelope clause (rev.1 item 7): exact two-sided binomial test of `n_OUT` against the **nearest** envelope edge, α=0.05 | `nearest_edge = min(SCATTER_ENVELOPE, key=lambda e: abs(f_out-e))`; `stats.binomtest(n_out, n_total, p=nearest_edge, alternative="two-sided")`; `envelope_ok = pvalue >= 0.05` | exact match to the draft's corrected §5 G-4 wording |

### 2b. Disposition table — every draft §4 row mapped to its code branch

| draft row | trigger | code branch | match |
|---|---|---|---|
| IMMATERIAL-FLOOR-SHARE | `|Δh_1D|<T_mat AND φ_1D<0.2 AND M_1D≥3` | `evaluate_dispositions`: `immaterial = "TRUE" if finite(d1,phi1,m1) and abs(d1)<t_mat and phi1<0.2 and m1>=3 else ...` | exact |
| CONE-OWNS-FLOOR | `\|Z_1D\|>3 AND φ_1D≥0.5 AND M_1D≥3` | `cone_owns = "TRUE" if finite(z1,phi1,m1) and abs(z1)>3 and phi1>=0.5 and m1>=3 else ...` | exact |
| INTERMEDIATE-UNPOWERED | `SE_1D > T_mat/3` | `unpowered = "TRUE" if finite(se1) and se1>t_mat/3 else ...` | exact — correctly independent of Δh/φ, per the draft's "whatever Δh and φ read" |
| INTERMEDIATE | `M≥3 AND ((\|Z\|>3 & 0.2≤φ<0.5) or (\|Δh\|≥T_mat & φ<0.2) or 1D/2D-disagree or leave-out-disagree>2SE)` | `intermediate_condition` (lines ~854-861) | formula structure matches; **see Finding D** on the "1D/2D disagree" operationalization |
| INSTRUMENT/NO-READ | G-1…G-4 red; g-population red → nothing banked | `main()`: `if not gates["passed"]: json.dump({"verdict":"INSTRUMENT-DEFECT", ...}); raise SystemExit(...)` **before** any statistic is computed | G-1..G-4 clause: exact match (hard stop, no banked statistic). g-population clause: **see Finding C — unreachable** |

### Findings (all non-blocking at the frozen launch configuration; none makes the registered read
wrong or unregistered as currently launched — hence overall GREEN, not RED)

**Finding A — `np.nansum` on `s_out - s_bar_in` silently zero-fills, rather than excludes, a
NaN-scored OUT event (medium, unconfirmed to manifest).** `stencil_scores` sets `s_e = NaN`
when either stencil-endpoint likelihood is `≤ 0` (the `ok = (lo>0)&(hi>0)` guard). If any
OUT-class event hits that guard, `np.nansum(s_out - s_bar_in)` treats its term as `0` in the
sum — not as excluded — while `n_out` (used in `SE` and reported) still counts it as a full
member of the OUT class. This differs from the T0 scorer's own physics-floor convention
(explicit floor-then-log, with an explicit exclusion mask), and from `_load_matrix`'s ragged-CSV
guard in `tier0_bootstrap_jackknife.py` (`raise ValueError` on any missing cell) — this script
has no equivalent raise. **Checked, not confirmed to fire:** the production diagnostics CSV is
a fully dense (event×h) grid with zero missing cells (`1588 events × 41 h = 65108 rows`,
`isna().sum().sum() == 0`, verified — the one aggregate check disclosed above), which rules out
one NaN source (missing rows); whether any of the 76 in-catalogue rows' `combined_no_bh`/
`combined_with_bh` values are exactly `≤ 0` at the two stencil `h` points was **not** checked
(would require reading values across the population, which this review declined to do).
Recommendation for the runner: log `n_out - len(s_out[np.isfinite(s_out)])` before summing, or
switch to an explicit exclude-and-report pattern matching `physics_floor_apply`'s convention.

**Finding B — `load_catalogue(k)` passes the sky-cone multiplier positionally as the handler's
`z_max`, not `k` (medium, currently harmless by numeric coincidence).**
`GalaxyCatalogueHandler.__init__(self, M_min, M_max, z_max, observed_catalogue_path=None)` —
the third positional argument is `z_max` (a catalogue redshift-pruning cutoff), **not** a
sky-cone multiplier. `load_catalogue(k: float)` calls `GalaxyCatalogueHandler(1e4, 1e7, k)`,
i.e. it feeds `args.sky_cone_k` (the registered `--sky-cone-k` CLI flag, a cone-radius
multiplier that is a genuinely different physical quantity) into the constructor's `z_max` slot.
The precedent this script cites (`cmem_a1.py:104`, `GalaxyCatalogueHandler(1e4, 1e7, 1.5)`)
hardcodes `1.5` as `z_max` directly — it does **not** parameterize `z_max` by any cone-related
`k`; the coincidence that the registered `--sky-cone-k` default (1.5) numerically equals the
intended `z_max` (1.5) is what makes today's launch (§7, "zero fresh choices") produce the
identical galaxy population as the precedent. But the coupling is a latent defect: any future
run of this same script with a **different** `--sky-cone-k` (a plausible robustness variant,
since the draft frames the flag as "generalized" per the module docstring) would silently also
change the catalogue's `z_max` pruning cutoff — an unregistered, undocumented side effect with
no warning. **Does not affect the registered read as launched** (§7's CLI is frozen at
`--sky-cone-k 1.5`), so this is not RED, but should be fixed before any sensitivity/robustness
variant of this launch: `load_catalogue` should hardcode `z_max=1.5` (matching `cmem_a1.py`)
independently of the `k` argument, which should only reach `cone_radius`.

**Finding C — the INSTRUMENT/NO-READ row's "g-population red" clause has no implementable
criterion, in either the draft or the code (low-medium, a registration-level gap more than a
code bug).** Draft §5 lists `g-population` purely as a disclosure ("harness 0 mixed rows
(`--population 200`...); production single pool") with no stated pass/fail threshold — unlike
G-1..G-4, which each carry an explicit "STOP on mismatch" / "Mismatch ⇒ INSTRUMENT-DEFECT"
consequence. `run_gates()`'s `gates["g_population_disclosure"]` dict correspondingly carries no
`"passed"` key and is not included in the `gates["passed"]` aggregate — meaning the disposition
table's "g-population red" trigger can **never** fire through this code path, regardless of what
the harness root actually contains. This is not a code defect against a specified formula (the
draft never specifies one), but it is a table clause with no reachable code path, which the
review's own criterion (2) flags. Recommendation: either the draft states a concrete
g-population criterion (e.g., "seed_S dir count must be in `{expected range}`") and the code
gates on it, or §4's INSTRUMENT/NO-READ row drops the "g-population red" clause and the
disclosure stays purely informational as §5 already frames it.

**Finding D — the INTERMEDIATE row's "1D/2D disagree in disposition" clause is an interpretive
operationalization, not a formula the draft states explicitly (low).** The draft's prose gives
no formula for "1D/2D disagree"; the code defines `disagree_1d_2d = ((|Δh_1D|≥T_mat) !=
(|Δh_2D|≥T_mat)) or ((φ_1D≥0.2) != (φ_2D≥0.2))`, i.e., a materiality-flag/φ-band mismatch. This
is a reasonable, literal reading of "disagree in disposition" and does not contradict anything
registered, but it is the builder's choice of formalization rather than a verbatim-registered
formula — worth a line in the runner's readout so the author can confirm it matches intent.

## 3. Cone synthetic-table check — verified by hand arithmetic (independently reproduced, GREEN)

### 3a. `cone_bias_floor_statistic`, 1D channel (`BUILD_RECORD.md` "FIX 2" 10-row synthetic table)

Reproducing the record's own numbers from its stated inputs (OUT = {0,1,2}: s=0.30,0.35,0.25;
IN = {3..9}: s=0.10,0.12,0.11,0.09,0.50,0.10,0.08):

- `s̄_IN` = (0.10+0.12+0.11+0.09+0.50+0.10+0.08)/7 = 1.10/7 = **0.1571428571** — matches
  `0.15714285714285717` ✓
- IN devs from median 0.10: {0, 0.02, 0.01, 0.01, 0.40, 0, 0.02} → sorted {0,0,0.01,0.01,
  0.02,0.02,0.40} → MAD (4th of 7) = **0.01** → `SD_IN` = 1.4826·0.01 = **0.014826** — matches
  `0.014826000000000013` ✓
- Two-outlier top-2 by `|s_e−median|`: event 7 (dev 0.40) then, among the 0.02-tie (events 4,
  9), event 9 — re-derived independently via `np.argsort(dev)` stable-tie behaviour on this
  input and confirmed to select `{7, 9}`, matching the record ✓
- `Δh_cone` = [(0.90) − 3·0.1571428571]/3256 = 0.4285714/3256 = **1.31625e-4** — matches
  `0.0001316251316251316` ✓
- `SE` = 0.014826·√(3+9/7)/3256 = 0.014826·2.070197/3256 = **9.4265e-6** — matches
  `9.42651595467729e-06` ✓
- `Z` = 1.31625e-4/9.4265e-6 = **13.963** — matches `13.963285296283964` ✓
- `φ` = 1.31625e-4/(−0.0630) = **−0.002089** — matches `-0.0020892878...` ✓
- `M` = 0.008/9.4265e-6 = **848.67** — matches `848.6698625944111` ✓

All eight numbers independently re-derived from the record's own stated inputs and confirmed to
full precision. The 2D-channel figures in the record follow the same arithmetic pattern
(spot-checked `s̄_IN,2D` = (0.05+0.06+0.055+0.045+0.20+0.05+0.04)/7 = 0.50/7 = 0.0714286,
matching the record).

### 3b. `t0_mean_h` / `physics_floor_apply` — independently reproduced on a NEW self-built
synthetic table (not the record's own, since the record's full 4×4 input table for that check
was not fully given in the excerpt read; this instead validates the *formula*, run on a table
this review built)

3-event, 2-`h`-value table with one deliberately-zeroed cell (event 2 at h=0.71):

| event | h=0.70 | h=0.71 |
|---|---|---|
| 0 | 0.20 | 0.30 |
| 1 | 0.10 | 0.15 |
| 2 | 0.05 | 0.00 → floored to 0.05 |

Hand computation: `logL` = `[[ln0.20,ln0.30],[ln0.10,ln0.15],[ln0.05,ln0.05]]`, `Σ_e logL` per
`h`, `w=np.gradient([0.70,0.71])=[0.01,0.01]`, `post_n = softmax(Σlog − max)·w`-normalized,
`mean_h = Σ(post_n·h·w)`. Hand/independent-script value: **`mean_h = 0.7069230769230769`**.
Running `t0_mean_h` from the actual `cone_loss_reads.py` module against a CSV built from this
exact table returns `{'mean_h': 0.7069230769230769, 'n_events_used': 3,
'n_events_floor_excluded': 0}` — **exact match to 16 significant figures**, and the physics
floor correctly replaced the zero with the row's own min-nonzero (0.05) without excluding the
row (only an all-zero row triggers exclusion, confirmed by code reading, matches
`tier0_bootstrap_jackknife.py`'s `_physics_floor_apply` semantics verbatim). `t0_mean_h` with
`exclude_event_idx={0}` correctly drops event 0 and recomputes on {1,2} only
(`mean_h=0.706`, `n_events_used=2`) — confirms `leave_out_cross_check`'s exclusion path is
wired correctly to `t0_mean_h`'s `exclude_event_idx` parameter.

Both the primary statistic and the leave-out T0-scorer formula are independently confirmed
correct by hand/script arithmetic.

## 4. Gates + STOP consequences

| gate | draft consequence | code | present |
|---|---|---|---|
| G-1 catalogue md5 | STOP on mismatch | `gates["g1_catalogue_pin"]["passed"]`, folded into `gates["passed"]` | yes |
| G-1 CRB md5 | STOP on mismatch | `gates["g1_crb_pin"]["passed"]`, folded in | yes |
| G-1 git commit (both venues) | STOP on mismatch | `gates["g1_git_commit_pin"]["passed"]`, folded in | yes |
| G-2 double anchor | "A miss = INSTRUMENT-DEFECT" | `gates["g2_passed"]`, folded in | yes |
| G-3 join | "Mismatch ⇒ INSTRUMENT-DEFECT" | `gates["g3_join"]["passed"]`, folded in | yes |
| G-4 KS clause | STOP/fresh RULE on failure | `gates["g4_scatter_law"]["ks_passed"]`, folded into `["passed"]` | yes |
| G-4 envelope clause (binomial, rev.1) | STOP/fresh RULE on failure | `gates["g4_scatter_law"]["envelope_passed"]`, folded in | yes |
| g-population | disclosure only in §5 (no stated STOP) but named in §4's INSTRUMENT/NO-READ row | not folded into `gates["passed"]`; no `"passed"` field exists for it | **gap — Finding C** |
| g-censoring, g-precision | narrative disclosures, no runtime check specified | not implemented as runtime gates | consistent — draft does not ask for a runtime check here |

The aggregate `gates["passed"] = g1_catalogue AND g1_crb AND g1_git_commit AND g2 AND g3 AND
g4` gates `main()`'s hard stop (`raise SystemExit(...)`) before any statistic is computed for
every G-1..G-4 failure mode — confirmed by reading `main()` and by `BUILD_RECORD.md`'s two
dry-run transcripts (pre-fix: G-4 envelope RED → top-level `passed: false`, reported
`INSTRUMENT-DEFECT`, exit 0 per the `--dry-run` contract; post-fix: all GREEN).

## 5. Launch block vs CLI — zero fresh choices, exact match

Every flag in `REGISTRATION_DRAFT.md` §7's launch command (`--production-crb`, `--production-run`,
`--replicate-run`, `--harness-root`, `--population`, `--anchor-fleet-mker`, `--anchor-fleet-cmem`,
`--sky-cone-k`, `--h-lo`, `--h-hi`, `--h-true`, `--crb-md5`, `--catalogue-md5`, `--out`,
`[--dry-run]`) has a corresponding `ap.add_argument` in `cone_loss_reads.py`'s `main()` with a
matching name and (where the draft gives one) matching default/registered value (`--sky-cone-k`
default `1.5`, `--h-lo` `0.725`, `--h-hi` `0.735`, `--h-true` `0.73`, `--git-commit` default
`"1ec9514d"` matching the draft's row #302 commit prefix). No CLI flag exists in the script that
is absent from the draft's launch block, and no flag in the launch block is missing from the
script. Confirmed line-for-line — no fresh choices.

## 6. Kill criterion / max_revisions / blindness-status

- **Kill criterion, verbatim check against the charter source** (not a forbidden file):
  `RESEARCH_GRAPH_1_PROPOSAL_20260901.md:46`, `q-cone-loss` row, quoted text: *"measurement
  confirms the floor within its registered uncertainty band -> settled as irreducible geometry;
  no fix pursued"* — **byte-identical** to the draft §4b quotation. ✓
- **max_revisions 2**: charter table row for `r-cone-loss` (line 157) states "max_revisions 2
  ORCHESTRATOR-DERIVED (provisional default, ratified with the charter...)" — matches the
  draft header's "max_revisions 2 (ORCHESTRATOR-DERIVED...)" and the arm cap "≤ 20 CPU-h" also
  independently confirmed at the same charter line. ✓
- **Blindness-status line**: present verbatim in draft §4b ("primary statistic point estimates
  exist in a gate record dated 2026-09-03 (unblinded by a design-gate side effect:
  DESIGN_GATE_stats.md); ... the registered read is executed by an agent that has not opened
  that record. The revising author did not open it."), plus the stage-0 fraction-disclosure
  sentence. Present as required; this reviewer independently did not open `DESIGN_GATE_stats.md`
  either (see header disclosure). ✓

---

## Summary for the author

Computability: GREEN — every input file and column the registered statistic needs exists on
disk with the schema the code expects. Formula-match: GREEN across the primary statistic
(Δh_cone, SE, Z, φ, M — all eight synthetic-table numbers independently re-derived to full
precision), the T0 leave-out scorer (independently re-implemented and matched to 16 s.f.), the
G-1..G-4 gates (all STOP-consequential, folded into one hard-stop aggregate), the launch CLI
(zero fresh choices, line-for-line), and the kill criterion/max_revisions/blindness-status
(all verbatim-matched against the charter, a non-forbidden source). Four non-blocking findings
are recorded for the runner/author: (A) a silent-NaN-to-zero risk in the OUT-class sum, not
confirmed to fire on the dense production grid; (B) a real but currently-harmless parameter
conflation in `load_catalogue` (`--sky-cone-k` doubles as the handler's `z_max`) that should be
decoupled before any robustness variant of this launch changes `--sky-cone-k`; (C) the
"g-population red" disposition clause has no implementable criterion in either the draft or the
code; (D) the "1D/2D disagree" clause is a reasonable but non-verbatim operationalization. None
of the four makes the registered read, as launched with the frozen §7 CLI, wrong or
unregistered — hence GREEN, not RED.
