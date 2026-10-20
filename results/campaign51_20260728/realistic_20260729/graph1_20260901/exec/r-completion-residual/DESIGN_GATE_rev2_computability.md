# DESIGN_GATE_rev2_computability.md — r-completion-residual, FRESH computability + formula-match re-gate

Reviewer: fresh agent, no prior DESIGN_GATE_*.md opened (rev1, design, provenance, stats all
untouched by this review; also did not open INFORMATION_FORECAST.md or cone_loss_result.json).
Inputs read: `REGISTRATION_DRAFT.md` (rev 1 + rev 1b), `completion_residual_reads.py`,
`BUILD_RECORD.md` (FIX 2 section), `MUSTFIX_REVISION1_20260903.md` (the must-fix list the draft
claims to answer), `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` (kill-criterion / max_revisions
verbatim source only). No production pipeline run, no cluster command, no edit under
`darksiren_emri/`. No aggregate computed over any registered population — every number below is
either read from `BUILD_RECORD.md`'s own printed dry-run JSON, hand arithmetic on ≤5 fabricated
rows, or a file/column/key existence check (`head`, `find`, `grep`, `md5sum`, `wc -l`).

**Overall verdict: RED.** Two registered-statistic/gate items named in §2.4/§5 of the draft have
no code path (δh_M; class closure), and the g-znorm gate is materially narrower than what §5
registers ("in both venues and in every harness universe"). None of these block the currently
GREEN headline discrimination (T_harn/Z_harn is correctly matched-channel, per-universe — see
check 3), but as registered the read is not fully self-defending: a red state in the harness data
that feeds T_harn/Z_harn (the statistic every disposition row keys on) has no gate to catch it.

---

## 1. Input existence (files / columns / JSON keys) — GREEN

Independently re-opened (headers/keys only, ≤5 rows, no aggregate):

| input | check | result |
|---|---|---|
| production `event_likelihoods.csv` | header | has `B_num, D_tilde_phi, alpha_G_phi, den_log_term, num_log_term_no_bh, combined_no_bh, event_idx, h` — every column §2.1/`t0_mean_h` needs |
| production `prepared_cramer_rao_bounds.csv` | header + md5 | has `host_galaxy_index`; `md5sum` reproduced independently = `9a1f2a14384a9281c97ca3be312ddaab` (matches CLI `--crb-md5` and `BUILD_RECORD.md`'s dry-run `match: true`); `wc -l` = 1591 (1590 rows), matches `n_crb_rows` |
| replicate `event_likelihoods.csv` (joint_r1) | header | identical column set to production |
| harness root | `find`/`ls` count | 67/67 checkpoints (`universe_seed*_S.json`), 67/67 `event_likelihoods.csv`, 67/67 `prepared_cramer_rao_bounds.csv` under `seed9010{00..66}_S/simulations/...` — re-verified independently, matches `BUILD_RECORD.md`'s FIX-2 pre-check |
| harness per-universe CSV/CRB | header, spot-checked seeds 901000, 901010, 901033, 901066 | same column set as production; `host_galaxy_index` present in every sampled CRB |
| harness checkpoint JSON | key dump, seed 901000 | `score_at_truth.no_bh.dark.{n,mean,sem}` present; `resolved_flags` has exactly 13 keys (`normalization_mode, catalogue_global_selection, selection_in_completion_numerator, catalogue_numerator_survival, catalogue_numerator_survival_2d, mass_filter_sigma, mass_filter_geometry, mass_filter_k, theta_b, theta_s, theta_sites, theta_phi_divisor, theta_zwindow`) — matches the draft's "resolved flags 13 tokens" |

No missing field anywhere the statistic touches. **GREEN.**

---

## 2. Statistic / SE / ρ / disposition-table formula match

### 2a. Registered statistics (§2.4) vs. code

| symbol | draft formula | code | match |
|---|---|---|---|
| T_prod | mean_e s_M,e (dark, production) | `t_prod = dark_terms["s_M"].mean()` | GREEN |
| SE_prod | SD_e(s_M,e \| dark, production)/√1512 | `dark_terms["s_M"].std(ddof=1) / len(dark_terms)**0.5`; `len(dark_terms)`=1512 confirmed by `n_dark_scored` in BUILD_RECORD dry-run | GREEN |
| Z_prod | T_prod/SE_prod | `t_prod/se_prod` | GREEN |
| T_harn | mean over universes of S_M,harn,U | `compute_harness_matched_channel_scores`: per-universe mean of dark `s_M` (own CSV, own `compute_event_terms` call), then `np.mean` across universes | GREEN — see check 3 below for the byte-id-vs-matched-channel distinction |
| SE_harn | SD_U(S_M,harn,U)/√67 | `np.std(values, ddof=1)/n**0.5`, `n`=number of *available* universes (67 when all present, which this run confirms) | GREEN in this run; **AMBER as a general formula** — code divides by √n_available, draft literally writes √67; only equivalent because g-population/byte-id independently forces n=67. Not wired together: nothing in `compute_registered_statistics` asserts `n_universes_available == 67` before trusting SE_harn. |
| Z_harn | T_harn/SE_harn | `t_harn/se_harn` | GREEN |
| ρ | T_harn/T_prod, only when \|Z_prod\|>3 | `rho = t_harn/t_prod if (abs(z_prod) > Z_BAND and t_prod) else None` | GREEN |
| δh_M (REPORTED-ONLY) | N_Ḡ·T_prod / I_1D, I_1D = 1/0.017526² = 3256 | **absent** — no `delta_h_M`/`I_1D`/`3256`/`0.017526` anywhere in `completion_residual_reads.py` (`grep` returns nothing); not in `compute_registered_statistics`'s return dict | **RED** — a table row with no code path. Named in the module docstring ("computes ... delta_h_M") and in §2.4's registered-statistics table, but never implemented. REPORTED-ONLY / not verdict-bearing, but it is still a registered output the launch block promises. |

### 2b. Disposition table (§4) vs. `compute_registered_statistics`'s `elif` chain

Verified by trace (each branch read against its literal trigger) and by the synthetic 6-triple sweep in `BUILD_RECORD.md`'s FIX-2 section, re-checked by hand:

| disposition | draft trigger | code branch | match |
|---|---|---|---|
| ILLEGITIMATE | \|Z_harn\|>3 AND ρ≥0.5 | `if abs(z_harn) > Z_BAND and rho is not None and rho >= RHO_ILLEGITIMATE` | GREEN |
| FLOOR-CONSISTENT | \|Z_harn\|≤3 AND \|Z_prod\|≤3 | `elif abs(z_harn) <= Z_BAND and abs(z_prod) <= Z_BAND` | GREEN |
| INTERMEDIATE (a) | \|Z_harn\|≤3 AND \|Z_prod\|>3 | `elif abs(z_harn) <= Z_BAND and abs(z_prod) > Z_BAND` | GREEN |
| INTERMEDIATE (b) | \|Z_harn\|>3 AND 0.2<ρ<0.5 | `elif ... RHO_MINOR < rho < RHO_ILLEGITIMATE` | GREEN |
| INTERMEDIATE (c) | \|Z_harn\|>3 AND ρ≤0.2 | `elif ... rho <= RHO_MINOR` | GREEN |
| INTERMEDIATE (d) HARNESS-ONLY-SIGNAL | \|Z_harn\|>3 AND \|Z_prod\|≤3 (ρ undefined) | `elif abs(z_harn) > Z_BAND and rho is None` | GREEN (this is the rev-1b addition, FIX 2 in BUILD_RECORD; confirmed present, `"unclassified"` fallback confirmed removed by `grep -c unclassified` = 0 in the file) |
| NO-READ | g-closure / JOIN / byte-id / g-population / g-znorm red | not a branch of `compute_registered_statistics` — realized as `run_dry_run`'s `gates_green` boolean, gating the human/agent launch decision, not a disposition value | see §3 below — **this is where the RED lives**, not in the six-way disposition chain, which is otherwise exhaustive (`else: raise AssertionError`, confirmed unreachable by the (Z_harn,Z_prod,ρ) case analysis — three regions of ρ (≥0.5, (0.2,0.5), ≤0.2) plus the ρ=None case are exhaustive and non-overlapping) |

Threshold constants (`Z_BAND=3.0`, `RHO_ILLEGITIMATE=0.5`, `RHO_MINOR=0.2`) match §3's frozen bands
exactly, and — per `MUSTFIX_REVISION1_20260903.md`'s explicit note — these were frozen before the
gate and untouched by revision 1; no drift found.

**Verdict for check 2: one RED (δh_M unimplemented), the six-way disposition chain and its
constants are GREEN.**

---

## 3. Cone arm / hand arithmetic — N/A (cone) / verified (completion)

No cone-arm files were provided to this review (only the completion arm's three files); the
cone-arm synthetic-table hand-check is out of scope here.

**Completion arm: is Z_harn built from per-universe matched-channel S_M, not the checkpoint
full-score mean?** Traced and confirmed **GREEN**:

- `compute_harness_matched_channel_scores` opens each universe's *own*
  `seed{seed}_S/simulations/diagnostics/event_likelihoods.csv` + sibling
  `prepared_cramer_rao_bounds.csv`, calls the identical `compute_event_terms` used for production,
  masks `host_galaxy_index == -1`, and averages `s_M` per universe — never touches
  `score_at_truth.no_bh.dark.mean` (the checkpoint's full-score aggregate).
- The checkpoint full-score mean is read only inside `reproduce_harness_byte_id`, whose output is
  wired into `T_full_harn_informational`/`SE_full_harn_informational` — explicitly labeled
  informational in the returned dict — and never into `t_harn`/`se_harn`/`z_harn`.
- Hand-verified the FIX-2 synthetic-table numbers in `BUILD_RECORD.md` against the stated
  per-universe values (0.013333…, 0.005, 0.02 for n_dark = 3, 2, 4):
  - T_harn = (0.013333333+0.005+0.02)/3 = 0.012777778 — matches the printed
    `0.012777777777782431` (analytic `0.012777777777777777`) to float precision.
  - SE_harn: deviations from the mean are +0.000555556, −0.007777778, +0.007222222; sum of squares
    ≈ 1.12963e-4; /(n−1=2) → variance ≈ 5.6481e-5; SD ≈ 0.0075154; SE = SD/√3 ≈ 0.0043390 —
    matches the printed `0.004339027597727321` (analytic `0.004339027597725920`) to 4 significant
    figures by hand, exact in the script's own float arithmetic.
  - Disposition sweep (6 triples) re-traced against the `elif` chain in §2b above: all six labels
    reproduce, including the Fix-2 row `Z_harn=5.0 Z_prod=1.0 rho=None -> INTERMEDIATE (d)`.

**GREEN.**

---

## 4. Gates + STOP consequences

§5's five named gates, checked against what the code actually computes and, separately, what it
actually *gates* on (i.e., feeds into a NO-READ-equivalent stop):

| gate | draft scope (§5) | code scope | verdict |
|---|---|---|---|
| g-population | harness (0 mixed rows, population tag, resolved-flag consistency) + production (1588 rows/h-node, JOIN gate, in-cat=76, dark=1512) | `check_production_population` runs on production + replicate; harness "population" check is only the checkpoint-tag filter inside `reproduce_harness_byte_id` (counts matched checkpoints) — **no row-count/JOIN-style check is ever run on the 67 harness per-universe `event_likelihoods.csv`/`prepared_cramer_rao_bounds.csv` pairs that actually feed T_harn** (only the checkpoint JSONs are population-gated, not the raw per-event CSVs `compute_harness_matched_channel_scores` reads) | **AMBER-leaning-RED**: the statistic-bearing files are ungated; only their checkpoint siblings are |
| g-closure | §2.1 identity ≤1e-9; **class closure** S_all = π_G·S_G + π_Ḡ·S_dark exact; red STOPs the read | `check_gclosure` implements the per-event identity only, called **only on the production `terms`** (dry-run and real mode both) — never on any of the 67 harness universes' `compute_event_terms` output, even though that output is exactly what `compute_harness_matched_channel_scores` averages into T_harn. **Class closure (S_all = π_G·S_G + π_Ḡ·S_dark) has no code path anywhere** — no `S_all`, no `pi_G`/`pi_Ḡ` weighting, confirmed by `grep` returning nothing. | **RED** (class closure absent) + **RED** (per-event closure never checked on harness data) |
| g-precision | full-precision cols for s_M/s_C; selection-table cross-check "where available" | `check_gprecision` implemented, never called (disclosed no-op in `BUILD_RECORD.md`: no `selection_tables_h_0_725.json`/`_0_735.json` exists) | GREEN as a disclosed conditional skip — matches "where available" |
| g-censoring | rail-fraction disclosure rule on any h-space quote | narrative-only requirement on the *draft's prose* (δh_M, §3.4), not a script output; N/A to this script (also moot since δh_M itself is unimplemented — §2a) | N/A |
| g-znorm | "at every h-node... in **both venues and in every harness universe**; any nonzero ⇒ g-znorm red ⇒ NO-READ" | `check_gznorm` is called on production (`g_znorm_production`, wired into `gates_green`) and on replicate (`g_znorm_replicate`, computed but **never referenced in `gates_green`** — a replicate-venue red would not flip the printed "gates all green" line) — and is **never called on any of the 67 harness universes** at all, not even informationally | **RED** — the registered gate's own scope ("both venues and in every harness universe") is only ~1/3 implemented (production only, and even the computed replicate half is dead code w.r.t. the gate decision) |
| g-byte-id | 67/67 harness dark full-score means bit-for-bit; T0 mean_h anchor | `reproduce_harness_byte_id` + `t0_mean_h`, both wired into `gates_green` | GREEN (independently re-confirmed: 67/67 checkpoints/CSVs/CRBs present, §1) |

STOP mechanics: `run_dry_run`'s `gates_green` boolean is printed to stderr but the script still
exits 0 regardless of its value (the STOP is procedural — a human/agent reads the line — not a
non-zero exit or exception). That is an acceptable design given the launch block says "Launch
waits on the byte-id gate GREEN... stamped GREEN in BYTEID_RECORD.md," i.e. the STOP is exercised
by the launch decision, not the script. **Real mode (`compute_registered_statistics`) is more
exposed**: it never calls `check_production_population` or `check_gznorm` at all (only
`check_gclosure` on production and the byte-id reproduction are present in its `"gates"` output
field), and it never checks any gate's boolean before computing and returning a disposition — a
red gate state at real-mode invocation time has no code path to a STOP; the whole design leans on
"a green dry-run was already run, on the same files, by someone else, earlier." That is consistent
with standing rule 2's two-agent handoff but is not self-defending inside the real-mode function
itself. **AMBER.**

---

## 5. Launch block — GREEN

`build_parser()`'s eleven flags (`--production-csv --production-crb --replicate-csv
--harness-root --population --h-lo --h-hi --h-true --crb-md5 --catalogue-md5 --out --dry-run`)
match REGISTRATION_DRAFT.md §7's launch block token-for-token, in the same order. No fresh choice
appears in the CLI that isn't already a named constant/flag in the draft (h-stencil, h-true,
population, both md5 pins, and the four file paths are all pinned in §7/§5 invariants, not chosen
at launch time).

---

## 6. Kill criterion, max_revisions, blindness-status line — GREEN

- Kill criterion: `RESEARCH_GRAPH_1_PROPOSAL_20260901.md:45` (q-completion-residual row) reads
  verbatim: *"registered arm fails to discriminate at its registered band after revision 2 -> park
  bounded-undetermined with the measured bound"* — **exact character-for-character match** to
  REGISTRATION_DRAFT.md §8b's quote (re-confirmed by direct `sed -n '45p'` on the charter file).
- `max_revisions 2`: confirmed at charter line 146 (`r-completion-residual` row): "max_revisions 2
  ORCHESTRATOR-DERIVED (provisional default, ratified with the charter; see r-b82-s4's derivation
  sentence)" — draft's §0 item 3 header ("max_revisions 2 (ORCHESTRATOR-DERIVED, charter
  §1.7/§1.13)") is consistent; both §1.7 ("Branch G — dark-class completion-leg residual program")
  and §1.13 ("Bounded cycles") exist as headers in the charter at lines 132/206.
- Blindness-status line: present in §8b, matches `MUSTFIX_REVISION1_20260903.md`'s "Both" item
  near-verbatim, with an added filename (`DESIGN_GATE_stats.md`) and an added confirming sentence
  — a strengthening, not a drift.

---

## Findings summary (most severe first)

1. **[RED] g-znorm gate does not cover its own registered scope.** §5 registers the check "in both
   venues and in every harness universe"; the code gates only the production venue
   (`g_znorm_replicate` is computed but dead — never read by `gates_green`), and never computes
   g-znorm for any of the 67 harness universes at all. A nonzero `den_log_term` spread in the
   replicate CSV or in any harness universe currently has **zero code path** to a NO-READ, contrary
   to the disposition table's own NO-READ trigger list ("g-znorm red"). File:
   `completion_residual_reads.py`, `run_dry_run` (lines ~415-425, ~595-603) and the total absence
   of a per-harness-universe call to `check_gznorm`.

2. **[RED] g-closure is never checked on the 67 harness universes that generate T_harn/Z_harn**,
   the statistic every disposition row keys on — only the production venue's closure is checked
   (dry-run and real mode alike). Compounding this, **the "class closure" component of the g-closure
   gate (S_all = π_G·S_G + π_Ḡ·S_dark) has no implementation anywhere** — not for production, not
   for the harness — despite being named explicitly in §2.1's gate definition.

3. **[RED] δh_M (§2.4, REPORTED-ONLY) is a registered-statistics-table row with no code path.**
   Named in the module docstring and in §2.4's table (formula: N_Ḡ·T_prod / I_1D, I_1D = 1/0.017526²
   = 3256) but `grep` for `delta_h_M`/`I_1D`/`3256`/`0.017526` in `completion_residual_reads.py`
   returns nothing; it is absent from `compute_registered_statistics`'s return dict. It is
   REPORTED-ONLY / non-verdict-bearing per §0 item 1, so this does not corrupt any disposition, but
   the registered read as built cannot produce one of its own promised outputs.

4. **[AMBER] Real-mode statistics computation does not re-verify any gate before banking a
   disposition** — `compute_registered_statistics` never calls `check_production_population` or
   `check_gznorm`, and includes only `g_closure`/`g_byte_id_harness` (both production/checkpoint-
   scoped) in its own `"gates"` output. The design relies entirely on a temporally-separate,
   already-green `--dry-run` having been run on the identical input files by a different agent
   (standing rule 2) — consistent with the launch protocol's intent, but not self-defending if real
   mode is ever invoked against inputs that drifted after the dry-run.

5. **[AMBER] SE_harn's `/√67` is implemented as `/√n_available`.** Correct and safer in general
   (protects against a partial-population run), but nothing in `compute_registered_statistics`
   asserts `n_universes_available == N_HARNESS_UNIVERSES (67)` before trusting `SE_harn`/`Z_harn` —
   it silently computes on however many universes loaded, with no gate tying that count back to the
   registered n=67 population declaration (the 67-count check that exists is on the *checkpoints*,
   in `reproduce_harness_byte_id`, not cross-checked against `harn_matched["n_universes_available"]`).

No defect found in: input file/column/key existence (§1); the six-way disposition `elif` chain and
its frozen thresholds (§2b); T_prod/SE_prod/Z_prod/T_harn/SE_harn/Z_harn/ρ formulas themselves
(§2a, modulo finding 5's missing cross-check); the harness-matched-channel-vs-checkpoint-full-score
separation (§3, hand-verified); the launch CLI (§5); the kill criterion, max_revisions, and
blindness-status line (§6).

## Consequence

Per the task's rubric ("RED only for a defect that makes the read wrong or unregistered"): findings
1–3 are read-integrity gaps in the registered protocol itself (a registered gate or registered
statistic with no code path is, by definition, an unregistered read in that respect), not merely
stylistic — hence **overall RED**. None of them currently changes the disposition this run would
produce (byte-id and production gates are green, and the hand-verified synthetic check confirms the
core T_harn/Z_harn machinery is correct where it does run) — but as registered, the read cannot show
that a harness-side g-znorm or g-closure defect, or the promised δh_M line, would be caught or
produced. This returns to the author as fresh RULE material for d-completion-register, not a
launch-blocking defect in the currently-green numbers themselves.
