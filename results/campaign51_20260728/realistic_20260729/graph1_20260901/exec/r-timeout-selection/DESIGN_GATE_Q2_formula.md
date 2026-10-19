# DESIGN_GATE_Q2_formula — r-timeout-selection, `q-timeout-population-mismatch` (Q2) ONLY

Reviewer: FRESH formula/integration pass, independent of `DESIGN_GATE_Q2_computability_rev2.md` (that
gate's own scope was pins/schema/row-counts — "no registered Q2 aggregate ... was computed"; this pass's
job is different: does every registered **formula** have a **correct code path**, hand-verified against
`timeout_q2_reads.py` and `synth_check_q2.py`). Own enumeration built directly from `REGISTRATION_DRAFT.md`
(base + REVISION 1 + REVISION 2 + `CHAIR ERRATUM`) before opening the checklist table in `BUILD_RECORD_Q2.md`,
per the design-gate lesson ("enumerate every draft-named item before building"). `MECHANISM_NOTE.md` read
for the code trace it supplies (log message text, catch sites); `INFORMATION_FORECAST.md` **not opened**
(FORBIDDEN, honored). Q1 is out of scope; no Q1 row below. No registered Q2 aggregate (S2.1–S2.4, `w_b`,
`w_e`, `Δmean_h^{Q2}`, `σ'_h/σ_h`, any Spearman/permutation/Fisher statistic) was computed over the
registered 1588-event population by this review — only `--dry-run` (real inputs, no aggregate), the
existing `≤10`-row synthetic fixtures (re-run + hand-checked, not extended), and direct `grep`/`md5sum`/
pandas-header reads of pinned files, exactly the class of activity the launch instruction permits builders
and reviewers.

**Verdict: GREEN, with one open, non-blocking, REPORTED-ONLY finding (F1) for the chair.** Every item that
feeds the two disposition rows (Q2-S2.2, **Q2-S2.3 PRIMARY**) and every gate (`G-1`, `g-byteid`,
`g-closure(ii)`, `g-scope`, the T0-convention import, the INSTRUMENT-DEFECT contract) has a correct,
independently hand-verified code path. One REPORTED-ONLY sub-item (S2.1's p0/e0 information-map bins)
does **not** implement the draft's own §2 text ("the pinned quintile edges") — it computes fresh
within-run quantiles instead of digitizing against the pinned `seed61000_p0_edges`/`seed61000_e0_edges`.
This item gates nothing (S2.1 is REPORTED-ONLY throughout §4/§5) and does not touch S2.2, S2.3, or any
`g-byteid`/`g-closure` gate, so it does not block Q2 launch on its own terms — but the S2.1 p0/e0 table
should not be read as "the pinned edges" until fixed or the draft text is corrected. Detail in §5.

## 0. What this pass ran (blindness line)

```
$ uv run python timeout_q2_reads.py --dry-run <the 8 real §1-pinned inputs, own invocation, not copy-pasted>
... n_kept per bin (g-byteid): [0, 9, 1276, 303, 0]
... n_timeout SNR-stage per bin (g-byteid): [206, 302, 216, 81, 15]
dry-run OK (no aggregate computed)
$ echo $?
0
```
own `--dry-run` invocation, real time 2.3 s; exit 0; pins matched; per-bin counts bit-identical to the
launch instruction's target `[0, 9, 1276, 303, 0]`. Also run: (a) the same launch block with `--crb-csv`
pointed at a nonexistent path → `INSTRUMENT-DEFECT: G-1 pin: CRB CSV not found at ...`, exit 1, no `--out`
file written (dry-run); (b) real mode (no `--dry-run`) with `--influence-iiib` pointed at a nonexistent
path → `INSTRUMENT-DEFECT` printed, exit 1, `--out` JSON written with `"disposition": {"value":
"INSTRUMENT-DEFECT", ...}` — confirms §6's "missing-input path is a hard INSTRUMENT-DEFECT" for both
CLI modes, not a silent skip. (c) `uv run python synth_check_q2.py` re-run fresh: 40/40 PASS, exit 0. (d)
`ruff check` + `mypy` on both files: clean. (e) two independent hand-arithmetic fixtures, own numbers, not
the ones in `synth_check_q2.py` (§3–§4 below) — one for `s2_3_weights`'s renormalisation, one for
`_weighted_moments`/`Δmean_h`. (f) direct `grep` of `darksiren_emri/main.py` (current working tree, not
trusted from `MECHANISM_NOTE.md`'s quotes) for every log-message string the scorer's regexes match, and a
raw sample of a real timed-out-event log line to hand-check the `_M_RE` regex against the actual
`np.float64(...)`-wrapped params format.

## 1. Own enumeration — every Q2-scoped item named in the draft, mapped to code

| # | draft item (section) | code path | hand-verified this pass? |
|---|---|---|---|
| 1 | Scored population = `{0..1589} − {1203,1356}` (§1 pin row) | `scored_crb()` (filters `event_idx.isin(SCORED_EXCLUDED_EVENT_IDX)`); `SCORED_EXCLUDED_EVENT_IDX=(1203,1356)` | YES — real dry-run: 1590→1588 |
| 2 | M bins, pinned seed61000-native edges, detector-frame M (§2) | `digitize_M()` = `np.digitize(M, edges[1:-1], right=False)`; `load_bin_edges()` reads `seed61000_M_edges` only | YES — bit-identical `[0,9,1276,303,0]` on real data |
| 3 | p0/e0 bins: **the pinned quintile edges**, REPORTED-ONLY (§2) | `s2_1_information_map()`: `pd.qcut(df["p0"],5,...)` / `pd.qcut(df["e0"],5,...)` | **NO — see F1, §5.** `load_bin_edges()` never reads `seed61000_p0_edges`/`seed61000_e0_edges` from the pinned JSON (confirmed: `grep -c "p0_edges\|e0_edges" timeout_q2_reads.py` = 0) |
| 4 | Re-weighted T0 posterior, `ln post'(h)=Σ_e[w_e(ln L_e(h)+δ_e(h))]`, gradient-trapezoid weights, "`tier0` `_moments`" convention (§2) | `_weighted_moments()`: `weights=np.gradient(h_grid)`; `logpost=(w_e[:,None]*logL).sum(0)`; `biv._moments(logpost[None,:], h_grid, weights)` | YES — hand arithmetic §4 below, exact match |
| 5 | `Σ_e w_e = 1588` always (`g-closure`) | `s2_3_weights()` renormalisation + explicit re-check in `main()` (`abs(sum_w-N_SCORED)>1e-6`) + `s2_3_reweighted_posterior()`'s own re-check | YES — proved an algebraic identity, §3 below (not merely spot-checked) |
| 6 | `σ_lnDL=sqrt(delta_luminosity_distance_delta_luminosity_distance)/luminosity_distance` (§2) | `s2_1_information_map()`, one line | YES — literal transcription, column names confirmed present in real CSV header (dry-run loaded successfully) |
| 7 | `Ω=2π\|sin qS\|sqrt(C_qSqS·C_φSφS − C_qSφS²)` (§2) | `_sky_area()` | YES — `det=c_qsqs*c_phisphis-c_qsphis**2; 2π·\|sin qS\|·sqrt(max(det,0))`; the `max(det,0)` guard is a numerical-safety clamp, not a formula change (det is a real covariance-determinant quantity, ≥0 up to fp noise) |
| 8 | `d_e = sign(0.73−mean_h)·(−influence_2D_e)`, "identical to r-offset-subset §2" | `s2_2_influence_vs_M()`: `d_e = joined[d_e_col].to_numpy(...)` (no sign transform applied in this script) | YES, and this is the correct code path — see §6 (traced into `build_influence_vector.py`: the pinned `influence_2D`/`influence_1D` CSV **columns already store the signed `d_e`**, not the raw unsigned influence; re-applying the sign here would be a double-negation bug. Confirmed by reading `build_influence_vector.py:41-51,199,303-304` directly, not assumed) |
| 9 | S2.1: n/median/IQR of σ_lnDL, Ω, SNR, generation_time per M bin (+p0/e0 REPORTED) | `s2_1_information_map()._stats()` | YES for the M-bin table; NO for p0/e0 sub-bins (F1) |
| 10 | S2.1: Spearman(log10 M, ln σ_lnDL), 10k-perm p | `s2_1_information_map()` tail | YES — `spearmanr`, `rng.permutation`, p = `(sum(|perm|>=|rho|)+1)/(n+1)` (standard permutation-p formula, correctly avoids a zero p-value) |
| 11 | S2.2: ρ_S(log10 M, d_e) [gates], ρ_S(log10 M, \|d_e\|) [REPORTED-ONLY] (REVISION 1 F4) | `s2_2_influence_vs_M()`: `rho`/`rho_abs`; only `rho`(non-abs) feeds `disposition_s2_2` | YES — `disposition_s2_2()` reads only `p_perm_d_e`/`any_bin_holm_p_lt_0p05`, never `rho_log10M_abs_d_e_reported_only` |
| 12 | S2.2 top-k Fisher/Holm, k=82/94/72 (iiib_2d/iiib_1d/jr1_2d) | `FAMILY_K={"iiib_2d":82,"iiib_1d":94,"jr1_2d":72}`; `s2_2_influence_vs_M()` per-bin 2×2 `fisher_exact` + `_holm()` | YES — k-values transcribed exactly; `_holm()` hand-checked (monotone, dominates raw p, α=0.05 boundary correct on a 5-value fixture, re-run) |
| 13 | S2.3 PRIMARY: `w_b=share_pool,det(b)/share_kept(b)` over supported bins **{2,3} only** (REVISION 1 F2/F3, REVISION 2 F5), unit weight elsewhere, **ONE** global renormalisation | `s2_3_weights()`, `SUPPORTED_BINS=(2,3)` | YES — hand-derived + numerically reproduced, §3 |
| 14 | `share_pool,det(b)`, `share_kept(b)` renormalised to sum to 1 **over the support only** (REVISION 1) | `share_kept_support`/`share_pool_det_support` dict comprehensions, denominator = `n_kept[list(SUPPORTED_BINS)].sum()` / `n_pool_det[list(SUPPORTED_BINS)].sum()` | YES |
| 15 | `share_pool,det(b)` = pool a-rows, SNR≥20, M-bin share (§4 S2.3) | `det = a[a["SNR"]>=20]` where `a = pool[pool["stratum"]=="a"]` | YES — literal transcription |
| 16 | Same-size null: 1000 draws, `w_e` permuted over events, seed 20260904 | `s2_3_reweighted_posterior()`: `rng=np.random.default_rng(null_seed); w_perm=rng.permutation(w_e)` × `null_draws` | YES — `NULL_DRAWS_DEFAULT=1000`, `NULL_SEED_DEFAULT=20260904` |
| 17 | `T_null = max(0.002, 2·SD(Δ_null))` (§5) | `t_null = max(0.002, 2*sd_null)` inside `s2_3_reweighted_posterior()` | YES — literal transcription |
| 18 | Anchors `mean_h`(iiib 2D)=`0.6658540600`, `σ_h`=`0.018474739`; iiib 1D=`0.6669870586`; jr1 2D=`0.6671274830` (§1) | `ANCHOR_MEAN_H` dict, `ANCHOR_SIGMA_H_IIIB_2D` | YES — bit-transcribed from §1's "Anchors" line |
| 19 | Channels: `combined_with_bh` (2D), `combined_no_bh` (1D) (§2) | `FAMILY_CHANNEL={"iiib_2d":"combined_with_bh","iiib_1d":"combined_no_bh","jr1_2d":"combined_with_bh"}` | YES |
| 20 | S2.3 decomposition REPORTED-ONLY: `share_to(b)` of the **820** SNR-stage timeouts (+2 CRB-stage, reported); D1-gate share NOT-EVALUABLE (CHAIR ERRATUM) | `s2_3_decomposition()` | YES — `share_to_snr_stage_820` denominator is `N_TIMEOUT_SNR_STAGE_TOTAL=820` exactly, not 822; CRB-stage 2 kept as a separate `..._reported_only` block, never folded in |
| 21 | S2.4 REPORTED-ONLY: timeouts' (log10 M, p0, mu/M) vs kept, all **822** | `s2_4_scatter_summary()`: `np.concatenate([snr_stage_timeout_M, crb_stage_timeout_M])` | YES — uses all 822 (SNR+CRB stage), correctly distinct from item 20's 820-only denominator |
| 22 | Q2-S2.2 disposition (3-valued, fresh RULE) | `disposition_s2_2()` | YES — band logic transcribed exactly, hand-checked in `synth_check_q2.py` Part 2d (re-run, all 7 PASS) |
| 23 | Q2-S2.3 disposition (3-valued, fresh RULE, `T_mat=0.008`, ratio band `[0.80,1.25]`/`[0.95,1.05]`) | `disposition_s2_3()` | YES — band logic transcribed exactly, hand-checked in `synth_check_q2.py` Part 2e (re-run, all 8 PASS) |
| 24 | Mandatory p0-scope line on every disposition row (§5) | `MANDATORY_P0_LINE`, embedded in both `disposition_s2_*` | YES — string bit-identical to §5's quoted line |
| 25 | `g-byteid`: `n_kept=[0,9,1276,303,0]` (REVISION 2 F5), `n_timeout`(SNR-stage)`=[206,302,216,81,15]` (CHAIR ERRATUM); "Any miss = INSTRUMENT" | `gbyteid_gate()` — raises `InstrumentDefectError` on either mismatch | YES — hand-verified the raise path with a deliberately perturbed array (§6) |
| 26 | `g-closure(i)` residual, Q1-scoped, disclosed only (§6) — must NOT gate Q2 | `gclosure_i_gate()` — computed, printed/stored, **never** raises, **never** feeds a Q2 disposition | YES — confirmed by reading every call site: `main()` calls it once, stores `residual` in `report["meta"]`, no `if residual...` branch anywhere |
| 27 | Q1 out of scope — no `p_det`/`SimulationDetectionProbability`/pool-timeout-tally touch (§3, script docstring) | n/a (absence check) | YES — `grep -n "SimulationDetectionProbability\|p_det\|completion_mass_factor_g\|_g_sel\|precompute_completion_denominator" timeout_q2_reads.py synth_check_q2.py` matches only the docstring's own prose sentence, zero executable references |
| 28 | p0 axis out of scope by construction — REPORTED-ONLY only (D1 record) | `by_p0_bin_reported_only`, `p0_median` in `s2_4_scatter_summary`; no p0 input to `disposition_s2_2`/`disposition_s2_3`, `s2_3_weights`, or `s2_2_influence_vs_M` | YES — grep confirms `p0`/`p0_bin` never appear inside either disposition function or `s2_3_weights`/`s2_2_influence_vs_M` |
| 29 | Frozen T0 convention imported, cited not re-derived (§1 pin row, §3 Phase C spec) | `sys.path.insert(...); import build_influence_vector as biv`; `biv._moments`, `biv._md5`, `biv._load_matrix` all called, never reimplemented | YES — §6 below |
| 30 | `--dry-run`: loads, pins, schema, per-bin counts; no aggregate; exit 0 | `main()`'s `if args.dry_run:` branch — returns before any `s2_1_information_map`/`s2_2_influence_vs_M`/`s2_3_weights` call | YES — own invocation, §0 |
| 31 | Missing/mismatched input = hard INSTRUMENT-DEFECT, before any statistic, never silent | `InstrumentDefectError` + `_write_instrument_defect()`; every `_check_pin`/`_check_manifest`/schema/population-size check raises it | YES — own two adversarial runs, §0/§7 |

## 2. `g-scope` (no statistic on p0 bins other than S2.1/S2.4 REPORTED-ONLY rows)

Grepped every function that feeds `disposition_s2_2` or `disposition_s2_3` (`s2_2_influence_vs_M`,
`s2_3_weights`, `s2_3_reweighted_posterior`, `_weighted_moments`) for `p0`: zero hits. `p0` appears only
inside `s2_1_information_map` (REPORTED-ONLY dict key `by_p0_bin_reported_only`) and
`s2_4_scatter_summary` (`p0_median`, REPORTED-ONLY). **GREEN**, unaffected by finding F1 (F1 is a
within-REPORTED-ONLY-scope formula deviation, not a scope-fence breach — no p0 statistic reaches a
gated row either way).

## 3. Hand-verified: the renormalisation is an algebraic identity, not merely a numerical spot-check

Built an own 6-event fixture (independent of, but structurally similar to, `synth_check_q2.py`'s Part 2b):
1 event in bin 1 (unsupported), 3 in bin 2, 2 in bin 3; pool a-stratum SNR≥20 rows: 2 in bin 2, 4 in bin 3.

```
n_kept = [0,1,3,2,0]           n_pool_det = [0,0,2,4,0]
share_kept_support  = {2: 3/5=0.6,        3: 2/5=0.4}
share_pool_det_supp = {2: 2/6=0.3333...,  3: 4/6=0.6667...}
w_b = {2: 0.3333/0.6 = 0.555556,  3: 0.6667/0.4 = 1.666667}
events (bin): [1(bin1)->1.0, 1(bin2)->0.555556 x3, 1(bin3)->1.666667 x2]
raw sum = 1.0 + 3(0.555556) + 2(1.666667) = 1.0 + 1.666667 + 3.333333 = 6.0 == len(events)
```
Code (`s2_3_weights()`, called directly) reproduces every one of these numbers bit-for-bit, including
`sum_w_e_pre_renorm = 6.0` and `sum_w_e_post_renorm = 6.0` (renormalisation factor `len(events)/total =
6/6 = 1.0`, a no-op on this fixture).

**This is not a coincidence — it is a general algebraic identity of the registered `w_b` formula.** For
any bin `b` in the supported set `S` (size `N_S = Σ_{b∈S} n_kept(b)`, pool-det support total `M_S =
Σ_{b∈S} n_pool_det(b)`): `w_b(b) = [n_pool_det(b)/M_S] / [n_kept(b)/N_S]`, so the weight mass contributed
by bin `b`'s kept events is `n_kept(b)·w_b(b) = n_pool_det(b)·N_S/M_S`. Summing over `b∈S`:
`Σ_b n_pool_det(b)·N_S/M_S = (N_S/M_S)·Σ_b n_pool_det(b) = (N_S/M_S)·M_S = N_S` — exactly, independent of
the actual `n_pool_det`/`n_kept` values, as long as both `share_*` dicts are normalised over the same
support set (item 14 above). Unsupported-bin events each contribute exactly `1`, and every kept event is
in exactly one bin, so `Σ_e w_e_raw = N_S + N_U = N_total = len(events)` **always**, before the explicit
renormalisation line even runs. On the real 1588-event population this means `sum_w_e_pre_renorm` should
already equal `1588.0` (up to floating-point rounding) before the `* (len(events)/total)` step — the
renormalisation is a defensive identity-check/rounding-guard on the registered population, not a
substantive rescaling, and `g-closure(ii)` is therefore satisfied **structurally by the formula's own
construction**, not merely by the code's final multiply. This also explains, independent of
`BUILD_RECORD_Q2.md §3`'s own account, exactly why the pre-fix bug (hard-coded `N_SCORED=1588` in the
renormalisation numerator) was invisible on the real population (`1588/1588=1` there too) and only
surfaced on a smaller synthetic table (`1588/6 ≠ 1`) — confirmed independently, not merely accepted from
the build record's narrative.

## 4. Hand-verified: one `Δmean_h` on a fresh (own) fixture, against `_weighted_moments`/`biv._moments`

Own 2-event, 3-h-point fixture (not `synth_check_q2.py`'s), unequal weights (`w_e=[1,3]`, the shape a
reweighted-vs-unit-weight event pair would actually take):

```
h_grid = [0.70, 0.73, 0.76]         weights = np.gradient(h_grid) = [0.03, 0.03, 0.03]
logL[A] = [ 0, -1, -2]              logL[B] = [-2, -1,  0]           w_e = [1, 3]
logpost = 1*logL[A] + 3*logL[B] = [0-6, -1-3, -2-0] = [-6, -4, -2]
lp = logpost - max = [-4, -2, 0]           post = [e^-4, e^-2, 1] = [0.0183156, 0.1353353, 1]
norm = Σ post·weights = 0.03·(0.0183156+0.1353353+1) = 0.0346095
post_n = post/norm
mean_h = Σ post_n·h_grid·weights = 0.7555281127666260...
sigma_h = sqrt(Σ post_n·(h_grid-mean_h)^2·weights) = 0.0119472203265381...
Δmean_h (toy anchor 0.73) = 0.7555281... - 0.73 = +0.0255281127666260...
```
Ran `q2._weighted_moments(logL, h_grid, w_e)` (the actual function, `biv._moments` underneath, unmodified)
on this exact fixture: `mean_h = 0.755528112766626`, `sigma_h = 0.011947220326538143` — **bit-identical**
to the hand values above (`abs(hand-code) < 1e-15` on both). This confirms, independently of
`build_influence_vector.py`'s own `_score_venue_channel` cross-checks and of `synth_check_q2.py`'s Part
2c (which checks range/sign properties but not a closed-form value), that `_weighted_moments` implements
the gradient-trapezoid-weighted normalised-posterior-mean formula exactly as specified in §2, with no
off-by-`w_e`-indexing or off-by-normalisation error.

## 5. F1 — S2.1 p0/e0 REPORTED-ONLY bins do not use "the pinned quintile edges" (open, non-blocking)

`REGISTRATION_DRAFT.md` §2: *"p0/e0 bins: the pinned quintile edges, REPORTED-ONLY."* This reads in
parallel with the immediately preceding M-bin sentence ("the pinned seed61000-native edges (§1)") — i.e.
p0/e0, like M, should digitize against **fixed, pre-computed** edge values. `design_gate_bin_edges.json`
(the same file `--bin-edges-json` pins) in fact carries exactly such values —
`seed61000_p0_edges: [3.677, 11.339, 12.861, 14.486, 19.958, 87.225]` and `seed61000_e0_edges: [0.0503,
0.0815, 0.1089, 0.1392, 0.1699, 0.1997]` (both quoted verbatim from the file on disk this pass) —
frozen against the **union(kept, timeout)** population per the file's own `population_source` field.

`s2_1_information_map()` does not use them. It calls `pd.qcut(df["p0"], 5, labels=False,
duplicates="drop")` / `pd.qcut(df["e0"], 5, ...)` **on `crb_scored` alone** — i.e. it re-derives its own
quintile edges from the *scored-only* p0/e0 distribution at run time, a different population (1588 kept
events only, vs. the pinned edges' 1590+822 union) and a different edge-selection rule (data-driven
per-run quantile vs. a value frozen once, before any timeout rate was inspected — the same "frozen
BEFORE any timeout rate/count is computed" discipline `design_gate_bin_edges.json`'s own header states
for the M edges). `load_bin_edges()` confirms this is not merely an unused-variable oversight: it parses
only the `"seed61000_M_edges"` key out of the JSON — `grep -c "p0_edges\|e0_edges" timeout_q2_reads.py` =
**0**, the pinned p0/e0 values are never read into the process at all.

**Scope of the defect.** This affects only `s2_1_information_map()`'s `by_p0_bin_reported_only` /
`by_e0_bin_reported_only` dict keys (item 9 above) — REPORTED-ONLY per §4/§5 throughout. It does not
reach `disposition_s2_2`, `disposition_s2_3`, `s2_3_weights`, `s2_2_influence_vs_M`, any `g-byteid`/
`g-closure` gate, or `s2_4_scatter_summary` (which reports `p0_median` directly, unbinned, from the
`kept_1588` block — no `pd.qcut` there). Confirmed by grep (§2 above): no gated function references
`p0_bin`. **This does not block Q2 launch** — the PRIMARY statistic (S2.3) and the two disposition rows
have zero dependency on this table. It is filed as an open item because a REPORTED-ONLY table quoting
different bins than the ones the draft names as "the pinned edges" is a genuine, fixable
formula-vs-code mismatch, not a stylistic nit: a reader comparing S2.1's p0-bin table against, say,
`rd-timeout-bin-seed61000`'s own p0-binned tallies (which do use the pinned edges) would be comparing two
different partitions of the p0 axis without being told so. **Recommended disposition for the chair (not
launched by this gate):** either (a) a one-line code fix — `digitize` p0/e0 against
`seed61000_p0_edges`/`seed61000_e0_edges` the same way `digitize_M` does for M, before Q2 real mode is
ever run — or (b) an append-only registration correction restating §2's p0/e0 sentence as "ad hoc
within-run quintiles of the scored population, REPORTED-ONLY" if that is in fact the intended read. Zero
cost either way (REPORTED-ONLY, no re-launch of any gated statistic required).

## 6. T0 convention: imported, not re-derived (confirmed)

```
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r-offset-subset"))
import build_influence_vector as biv
```
Every place `timeout_q2_reads.py` needs the frozen T0 convention, it calls into `biv`, never
reimplements: `_load_logl()` uses `biv._md5` (pin check) and `biv._load_matrix` (physics-floor apply +
log; raises `InstrumentDefectError` if `n_excluded != 0`, per item "0 physics-floor exclusions expected"
in `REGISTRATION_DRAFT §6 g-precision`); `_weighted_moments()` uses `biv._moments` directly on the
`w_e`-summed `logpost`, with `weights=np.gradient(h_grid)` computed the same way `biv._score_venue_channel`
computes it (confirmed by reading `build_influence_vector.py:178-186` directly, not assumed). No formula
in `_moments` (the gradient-trapezoid normalisation, mean, variance) is duplicated or reimplemented in
`timeout_q2_reads.py` — grep for `def _moments`/`np.gradient` inside `timeout_q2_reads.py` finds only the
one `weights = np.gradient(h_grid)` line inside `_weighted_moments`, matching `biv._score_venue_channel`'s
own `weights = np.gradient(h_grid)` line verbatim. Also traced §2's `d_e` claim into `build_influence_vector.py`
directly (item 8 above) rather than trusting the draft's symbolic restatement — confirmed the pinned
`influence_2D`/`influence_1D` CSV columns already carry the signed `d_e`, so `s2_2_influence_vs_M`'s
un-transformed read of that column is the correct implementation of "identical to r-offset-subset §2",
not a missed sign flip.

## 7. INSTRUMENT-DEFECT path (own adversarial runs, both CLI modes)

| scenario | mode | result |
|---|---|---|
| `--crb-csv` → nonexistent path | `--dry-run` | `INSTRUMENT-DEFECT: G-1 pin: CRB CSV not found at ...`; exit 1; `--out` **not written** |
| `--influence-iiib` → nonexistent path | real (no `--dry-run`) | `INSTRUMENT-DEFECT` printed; exit 1; `--out` JSON **written** with `"disposition":{"value":"INSTRUMENT-DEFECT", "instrument_note": "...", "detail": {...}}` — a machine-readable halt, not a silent skip |
| deliberately wrong CRB md5 (3-row fixture) | real | `synth_check_q2.py` Part 1(a), re-run this pass: exit 1, `INSTRUMENT-DEFECT` printed, JSON `disposition.value == "INSTRUMENT-DEFECT"` — PASS |
| correct md5, wrong population size (3-row fixture, real CLI) | real | Part 1(b), re-run: exit 1, `INSTRUMENT-DEFECT` — PASS |
| same broken fixture, `--dry-run` | dry-run | Part 1(c), re-run: exit 1, `--out` never written — PASS |
| `gbyteid_gate` with a perturbed `n_kept`/`n_timeout` array | direct call | Part 2a, re-run: raises `InstrumentDefectError` naming `n_kept`/`n_timeout` respectively — PASS |

Confirms item 31: every missing/mismatched/wrong-shaped registered input is a hard, typed
`InstrumentDefectError`, raised before any `S2.x` statistic executes (checked by reading `main()`'s
control flow: all pin/schema/population checks happen inside the first `try:` block, before the
`if args.dry_run:` branch and before any `s2_1_information_map`/`s2_2_influence_vs_M`/`s2_3_weights`
call), never a silently-skipped/defaulted value.

## 8. Items independently cross-checked against the live working tree (not trusted from `MECHANISM_NOTE.md`)

`grep -n "computation timed out\|in dervative\|evaluations successful\|Caught ZeroDivisionError"
darksiren_emri/main.py` (current HEAD, not the draft's pinned `79c44608` — the physics-relevant lines are
unchanged; the intervening commits are docs-only, confirmed by `git log --oneline -5` showing only
`docs:`-prefixed messages since):

```
566:  f"{counter} / {iteration} evaluations successful. ..."
760:  "Caught ZeroDivisionError during trajectory integration. Continue with new parameters..."
768:  "Waveform/SNR computation timed out (>90s). Skipping event... params=%s"
799:  "Caught ParameterOutOfBoundsError in dervative. Continue with new parameters..."
815:  "Cramér-Rao bound computation timed out (>90s). Skipping event... params=%s"
```
`_SNR_STAGE_MSG`/`_CRB_STAGE_MSG`/`_Y_RE`/the `"in dervative"` substring check all match these lines
exactly (the CRB message check is the substring `"bound computation timed out"`, present in both the
ASCII- and accented-`Cramér` encodings, correctly encoding-agnostic). Pulled a real sample line from
`seed61000/cluster_logs_fetch_20260904/logs/simulate_6088772_16.err`:
`params={'M': np.float64(209475.60958010424), 'mu': 10, 'a': 0.98, 'p0': 12.055272845799983, ...}` — the
`_M_RE` pattern `r"'M':\s*(?:np\.float64\()?([\-0-9.eE]+)\)?"` matches this `np.float64(...)`-wrapped
format correctly (verified by inspection against the exact string, not merely by the aggregate parse
count already matching `820`/`2` in §0's dry-run).

## 9. Bottom line for the chair

Every code path feeding `Q2-S2.2` and `Q2-S2.3` (**the two rows §5 actually gates**, including the
PRIMARY `Δmean_h^{Q2}`/`σ'_h/σ_h` statistic and its same-size null), every `g-byteid`/`g-closure(ii)`/
`g-scope` gate, the frozen-T0-convention import, and the INSTRUMENT-DEFECT contract on every registered
input are correct, independently hand-verified against fresh (own) numbers and the live working tree —
not re-used from `synth_check_q2.py`'s or the prior computability gate's own fixtures/claims. One
REPORTED-ONLY item (S2.1's p0/e0 information-map bins, F1 §5) does not implement the draft's literal
"pinned quintile edges" text; it is scoped away from every gate and disposition and is a same-day,
zero-cost fix or a one-line registration correction, either way not a launch blocker. **Q2 is launchable
on its PRIMARY and disposition-gating statistics as registered; F1 is routed to the chair for a same-day
[DO] before the S2.1 REPORTED-ONLY table is read as "the pinned edges."**
