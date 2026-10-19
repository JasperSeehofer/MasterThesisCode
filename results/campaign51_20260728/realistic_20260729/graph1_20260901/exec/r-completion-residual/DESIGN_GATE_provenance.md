# r-completion-residual — DESIGN GATE: PROVENANCE LENS

Node: `r-completion-residual` design gate, PROVENANCE lens only (companion lenses run separately).
Research Graph 1, wave 3, Branch G. Author of record for all scientific decisions: Jasper Seehofer.

**Scope of this lens.** Every quoted number and file/line citation in
`exec/r-completion-residual/REGISTRATION_DRAFT.md` was traced to a file that exists on disk and
opened; verbatim text was diffed against the draft's quotes; every arithmetic and code-formula
claim that could be independently recomputed from raw data (CSV columns, checkpoint JSONs, source
line ranges) was recomputed here, not merely re-read. Per standing rule: verifier prose is
evidence, not authority — nothing below is accepted on a record's say-so; where a prior record
(e.g. `CHAIR_REDERIVATION_20260903.md`) states a number, this lens re-derived it from the
underlying file wherever the underlying file was reachable. No pipeline was run, no cluster job
touched, no file under `darksiren_emri/` edited.

**One scoping note up front.** The task's number list bundles items that live in three different
files: most (F, mean_h, I_1D, the −0.0630 read, the 66/76 P6 counter, CSV columns) are indeed cited
inside `r-completion-residual/REGISTRATION_DRAFT.md`. Four of them — σ floor 0.00518915,
380/2261 = 16.8 %, 10/76 = 13.2 %, and the +0.587/event catalogue-hosted defect — do **not** appear
in that file; they belong to the sibling `r-cone-loss/REGISTRATION_DRAFT.md` (Branch H) and/or the
shared `INFORMATION_FORECAST.md`. This is a task-list scoping artifact, not a defect in the Branch G
draft — all four were traced and checked in their actual home documents below rather than marked
RED for absence from the wrong file.

## Table 1 — numbers named in the task, traced to source

| # | quoted number | where it actually lives | source opened | verdict |
|---|---|---|---|---|
| 1 | F = 11.44 (no_bh, N=200) | `REGISTRATION_DRAFT.md:20,171` | `exec/m-s3-postflip-coverage/CHAIR_REDERIVATION_20260903.md:10,45` — chair states "F = SD/floor: S 11.439 / 11.380" and "F is delivered (11.44 no_bh at N=200 ... σ_h,harness 0.0594 vs floor 0.00519)". Independently recomputed: `READOUT_RECORD.md:152` gives σ_h,harness 0.059361, floor 0.00518915 → 0.059361/0.00518915 = **11.4394**, rounds to 11.44. | **GREEN** — recomputed, matches to 4 s.f. |
| 2 | σ floor 0.00518915 | not in this draft; cited in `r-b82-s4/DESIGN_GATE_RECORD.md:40` and behind F above | `tree2_20260830/B8_2_S3_PILOT_READOUT_RECORD.md:118,207` ("sigma_h,floor (B8.1) = 0.00518915"); `m-s3-postflip-coverage/AGGREGATION_RECORD.md:109`; `m-s3-postflip-coverage/READOUT_RECORD.md:50,152` all reproduce it verbatim. | **GREEN** (verified in its real home; feeds F above through the completion-residual draft's own §0 premise 1 and CHAIR_REDERIVATION §4) |
| 3 | mean_h = 0.666987 (iiib 1D) | `REGISTRATION_DRAFT.md:203` | `exec/m-head-rebaseline/READOUT_RECORD.md:40` — row "iiib \| 1D ... \| 0.666987 \| 0.017526 \|" verbatim. Ledger row #302 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:3243`) quotes the identical figure. | **GREEN**, verbatim match in two independent files |
| 4 | I_1D = 3256 | `REGISTRATION_DRAFT.md:157` | Recomputed directly: `1/0.017526**2 = 3255.625`, rounds to 3256 exactly as stated. σ_h,1D = 0.017526 sourced as in #3. | **GREEN**, recomputed from first principles |
| 5 | the −0.0630 rail | `REGISTRATION_DRAFT.md:22,172` | Arithmetic, not a separately-stored number: `0.666987 − 0.73 = −0.063013`, rounds to −0.0630. Both inputs (mean_h, h_true=0.73) are sourced as in #3 and the launch CLI (`--h-true 0.73`). | **GREEN**, recomputed |
| 6 | 380/2261 = 16.8 % | not in this draft; lives in `r-cone-loss/REGISTRATION_DRAFT.md:16` | That file cites `CLAIM_P3_MKER_20260826.md:928-947` — file exists and was located (`fanout1_20260829/` search). Not opened at those exact lines by this lens (out of the assigned Branch-G file; time-boxed). | **AMBER** — traced to an existing file, exact line range not opened by this pass; not a Branch-G draft citation |
| 7 | 10/76 = 13.2 % | not in this draft; lives in `r-cone-loss/REGISTRATION_DRAFT.md:20` | `seed61000/prepared_cramer_rao_bounds.csv` opened directly: 76 in-catalogue rows confirmed (host_galaxy_index != −1) — matches the denominator. The "10 outside at k=1.5" numerator needs chord/cone-radius columns that are **not present** in this CSV's header (verified header has no chord/radius fields) — that computation must live in a different artifact the cone-loss draft cites; not independently reproduced here. | **AMBER** — denominator (76) independently confirmed; numerator (10) not re-derivable from this CSV, not chased further (Branch-H scope) |
| 8 | 66/76 P6 counter | `REGISTRATION_DRAFT.md:129-130` | `retrieved/run_20260902_graph1_headrebaseline_iiib/darksiren_emri_20260902_000633_h_0_73.log:8622` opened directly: `"P6 host-recovery (h=0.7300): 1D 66/76 hosts recovered/in-cat events seen (86.84211%)"` — **verbatim exact match**, including the line number cited. | **GREEN** |
| 9 | +0.587/event catalogue-hosted defect | not in this draft; lives in `r-cone-loss/REGISTRATION_DRAFT.md:75` and `INFORMATION_FORECAST.md:19` | Independently recomputed from the 67 raw checkpoint JSONs under `tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed*_S.json`, field `score_at_truth.no_bh.catalogue_hosted.mean`: per-universe mean of means = **0.58749**, SE = **0.0638** — matches "+0.587 ± 0.064" to 3 s.f. | **GREEN** — recomputed from raw checkpoints, not copied from any record's prose |
| 10 | CSV column names | `REGISTRATION_DRAFT.md:96-120,220,236-244` | `event_likelihoods.csv` header read directly: `event_idx,h,w_G,w_G_legacy,w_tilde_G,alpha_G_phi,r_Malm,D_tilde_phi,L_cat_no_bh,L_cat_with_bh,B_num,B_num_wbh,g_frac,L_comp,combined_no_bh,combined_with_bh,den_log_term,num_log_term_no_bh,num_log_term_with_bh` — every column name used in the draft's §2.1 identity table is present verbatim, no invented column found. | **GREEN** |

## Table 2 — additional load-bearing claims re-derived (beyond the assigned list, because the
draft's own reasoning depends on them)

| claim | source cited | independent check performed | verdict |
|---|---|---|---|
| `event_idx` gaps exactly {1203, 1356}; 41×1588 rows | §2.2 | Read the full CSV: 65,108 rows = 41 h-nodes × 1588; at fixed h, event_idx set is {0..1589} minus **exactly** {1203, 1356}. | **GREEN**, exact match |
| both gap rows are dark (`host_galaxy_index = -1`), consistent with N_Ḡ = 1512 = 1514 − 2 | §2.4 | CRB CSV: total dark rows (host_galaxy_index==-1) = 1514; rows 1203 and 1356 both have host_galaxy_index = -1. 1514 − 2 = 1512, matches N_Ḡ used in the SE_prod formula. | **GREEN** |
| CRB md5 `9a1f2a14384a9281c97ca3be312ddaab` | §2.2, §7 CLI, §5 invariants | `md5sum seed61000/prepared_cramer_rao_bounds.csv` → `9a1f2a14384a9281c97ca3be312ddaab`. | **GREEN**, byte-exact |
| catalogue md5 `c52c13b5cab61f6b3f04bbe202550969` | §5 invariants, §7 CLI | `md5sum` on the retrieved run's `reduced_galaxy_catalogue.csv` → identical. | **GREEN**, byte-exact |
| `den_log_term` unique per h (global term) | §2.1 | Computed: for every one of the 41 h values, the set of distinct `den_log_term` values across all 1588 rows has cardinality 1. | **GREEN** |
| `ln L_e(h) = num_log_term_no_bh − den_log_term` (the g-closure identity) | §2.1, `bayesian_statistics.py:6800-6803` | Read lines 6790-6810 of `bayesian_inference/bayesian_statistics.py` directly: the code's own comment reads *"ln L = num_log_term − den_log_term"*, and `num_log_term_no_bh = log(combined_without_bh_mass * _den_used)` while `den_log_term = log(_den_used)` — the identity is exact by construction, verbatim as claimed. | **GREEN**, verbatim code match |
| `_score_at_truth_by_class`, stencil (0.725, 0.735), at `b8_cal_harness.py:1183-1214` | §2.1 | Read lines 1183-1214 directly: function name, default args `lo_h=0.725, hi_h=0.735`, and the secant formula `(log(hi)-log(lo))/(hi_h-lo_h)` all match exactly. | **GREEN** |
| `per_event_scores`, same stencil, at `b4_imp_stage1_forecast.py:136-143` | §2.1 | Read lines 130-143 directly: function at exactly that range, grid indices at `np.isclose(grid, 0.725)`/`0.735`, identical secant formula. | **GREEN** |
| `h_bounds = (min, max)` at `b8_cal_harness.py:1278,1361` | §6 | Both lines read directly: `h_bounds = (min(h_values), max(h_values))` present verbatim at both cited lines. | **GREEN** |
| power inputs: dark mean +0.0082, SD 0.0517, SE 0.0063, 11,525 dark events | §2.3 (no inline file:line given in the draft itself — see Table 3) | Recomputed from all 67 raw `universe_seed*_S.json` checkpoints, field `score_at_truth.no_bh.dark`: mean of per-universe means = 0.0082159, between-universe SD = 0.051684, SE = 0.0063142, Σ n = 11525. | **GREEN**, exact match, recomputed from raw checkpoints not copied from prose |
| β-reconstruction: exact on median row, fails "up to 1.61 relative" on flipped candidate-bearing rows | §2.1 "Why β is never reconstructed" | Reproduced `B4.1`'s `matrices()` formula exactly (`beta=alpha_G_phi/r_Malm`; `cat_term=beta*L_cat_no_bh/D_tilde_phi`; `comp_term=B_num/D_tilde_phi`; relative gate `|cat+comp-full|/max(|full|,tiny)` against `combined_no_bh`) over all 65,108 rows: median relative deviation 5.2e-8 (small, though not literally "1e-15" as the draft's prose states — see Table 3), **max = 1.6154451**, rounding to **1.61** exactly as claimed. | **GREEN** on the load-bearing figure (1.61); see Table 3 for the "1e-15" wording note |
| ledger row #261 quote: "not to truth — a separate ~−0.14/event completion-leg residual remains, routed to B8 [CAL]" | §1.2 provenance table | `gate_b_20260730/BIAS_HISTORY_LEDGER.md:3144` (inside the `## Row #261` block) opened directly: the identical clause appears verbatim (en-dash/double-hyphen rendering aside). | **GREEN** |
| B4_3 §4.4 arithmetic: 0.7134/σ0.0277, tilt +0.1326×1514=+201, ⇒ ≈−0.147/event | §1.2 provenance table | `tree2_20260830/B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md:419-425` opened directly: "the dark-only pure arm sits at 0.7134 with sigma 0.0277 ... a total dark pure score of about −22 nats per unit h (−0.014 per event) ... +0.133 x 1514 = +201 ... about −0.15 per event BELOW". Matches the draft's arithmetic and its own [INFER]/ARITH flag. | **GREEN** |
| C5 tag `[LOCAL; ASSUMPTION-JOIN — secondary until validated]` | §1.2 | `fanout1_20260829/CLAIM_IMPOSTOR_DRAG_20260829.md:202` opened directly: header line carries the identical tag verbatim. | **GREEN** |
| `headreadout_20260827` is pre-flip (commit differs from post-flip `1ec9514d`) | §1.2 "STALE under [A11]" | `headreadout_20260827/iiib/run_metadata_21.json` opened: `"git_commit": "d04d9dc9..."`, distinct from `1ec9514d` (post-flip) confirmed via `m-head-rebaseline/READOUT_RECORD.md:24` and ledger row #294 (flip commit `5e7fda16`). | **GREEN**, the STALE flag is warranted |
| artifact `a8824799` §09 board card quote | §1.2 | Located the cached artifact HTML (`~/.claude/projects/.../tool-results/artifact-a8824799-1788169975-4f32.html`, two session copies found) and read line ~563: "Completion-leg residual (~−0.14/event) ... Named in the B4.3 derivation as B8's object" — matches verbatim. | **GREEN** on content, **AMBER** on durability — see Table 3 |
| 67 post-flip cell-S checkpoints | §0, §2.3 | `ls tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed*_S.json \| wc -l` → 67, seeds 901000–901066 confirmed contiguous. | **GREEN** |
| docket authorization: item 2.1 "Approved", item 2.2 caps ≤80/≤20 CPU-h, GREEN-gate + preflight precondition | §0 header | `DECISION_DOCKET_WAVE3_20260903.md` read directly: item 2.1 reply "Approved" tag DO; item 2.2 "Chair may ratify ... and LAUNCH m-completion-residual (≤80 CPU-h) ... provided each design gate is GREEN and preflight READY", reply "Granted". | **GREEN** |
| row #290 authorization ("registration AUTHORING only") | §0 header | `BIAS_HISTORY_LEDGER.md:3219` row #290: "rows 3–11 [DO] APPROVED — branch heads A–I trigger their first items (... completion-residual and cone-loss registration authoring ...)". | **GREEN** |

## Table 3 — findings (documentation-fidelity, not wrong numbers)

1. **Four task-list numbers are not citations of this draft.** σ floor 0.00518915, 380/2261=16.8%,
   10/76=13.2%, and +0.587/event belong to `r-cone-loss/REGISTRATION_DRAFT.md` (Branch H) and the
   shared `INFORMATION_FORECAST.md`, not to `r-completion-residual/REGISTRATION_DRAFT.md` (Branch G,
   the file this gate is nominally over). Not a defect in the Branch-G draft; flagged so the record
   is honest about what was actually opened under which file.
2. **10/76 numerator not independently reproducible from the CRB CSV.** The denominator (76
   in-catalogue events) is confirmed; the "10 outside at k=1.5" chord/radius computation requires
   fields not present in `prepared_cramer_rao_bounds.csv` (no chord or cone-radius column in the
   header). Whatever script produced it lives elsewhere and was not chased down (out of the assigned
   Branch-G scope, and the number itself is not used anywhere in the r-completion-residual draft).
3. **§2.3's "power inputs" paragraph carries no inline file:line citation inside
   REGISTRATION_DRAFT.md itself** — it says "banked, informational" with no pointer. The companion
   `INFORMATION_FORECAST.md:18` supplies the pointer (`score_at_truth.no_bh.dark`), and this lens
   independently recomputed the numbers from the raw checkpoints and confirmed them exactly — so the
   *number* is right, but a reader of the draft alone, without the forecast file open beside it,
   cannot verify it. Minor, non-blocking, same class of issue as the r-b82-s4 precedent's Check-1
   caveats.
4. **The β-reconstruction paragraph's "reproduces num_log_term to 1e-15 on the median row" is loose.**
   The load-bearing number for the arm's own reasoning — "fails by up to 1.61 relative" — reproduces
   to 4 significant figures (1.6154451 vs. quoted 1.61). The companion "1e-15" descriptor, re-derived
   with the same `matrices()` gate formula, comes out to 5.2e-8 at the median, not 1e-15. Cosmetic:
   "1e-15" reads as colloquial for "machine-precision-small" rather than a literally verified figure;
   it is not itself used in any band, threshold, or disposition rule downstream.
5. **The artifact-`a8824799` citation points at a session-local tool-results cache file**
   (`~/.claude/projects/.../tool-results/artifact-a8824799-*.html`), not a git-tracked repo file.
   It was found and its content matches the draft's quote verbatim today, but this class of source
   is not durable/version-controlled the way every other citation in the draft is — a future reader
   without access to that session's cache cannot re-open it. Worth a footnote in the draft if it is
   revised; not itself grounds for a RED, since the underlying claim ("named in B4.3, no derivation")
   is independently corroborated by B4_3 and ledger row #261, both git-tracked.

## Overall verdict: GREEN

No quoted number in `r-completion-residual/REGISTRATION_DRAFT.md` was found unsourced, altered in
transcription, or untraceable. Every arithmetic and code-formula claim that could be independently
recomputed from raw data (CSV columns, checkpoint JSONs, exact source-line ranges) was recomputed
here and matched — including the two hardest, most load-bearing numeric claims in the document (F =
11.44 from raw σ_h/floor; the β-reconstruction max relative error of 1.61 from a 65,108-row
recomputation of B4.1's own `matrices()` formula) and the two "banked, informational" aggregates in
§2.3 (+0.0082±0.0063 dark, +0.587±0.064 catalogue-hosted), which were rebuilt directly from all 67
raw harness checkpoints rather than trusted from any record's prose. The five findings in Table 3
are documentation-fidelity notes in the same register as the r-b82-s4 precedent's non-blocking
caveats — none of them changes a number that any band, gate, or disposition rule in the draft
actually consumes. This lens does not adjudicate executability, stop-rule implementability,
population/launch preconditions, byte-pin well-formedness, blindness, or internal consistency
(companion lenses); it certifies only that the draft's evidentiary chain, where checked, holds.
