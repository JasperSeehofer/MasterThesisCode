# DESIGN_GATE_rev1_computability.md — r-completion-residual REGISTRATION_DRAFT.md revision 1

**Reviewer:** fresh computability-only design-gate agent (no prior gate record read; this is a first
read of the object). **Scope:** computability only — column/key existence, SE/formula match between
draft and script, disposition-table completeness, launch-block/CLI match, kill-criterion quote,
blindness-status line. Did not open `DESIGN_GATE_stats.md`, `INFORMATION_FORECAST.md`, or
`cone_loss_result.json`; did not run the production pipeline, cluster commands, or edit
`darksiren_emri/`; loaded at most 5 rows of any data file for column existence and computed no
aggregate over the registered population (means/SDs quoted below are values already sitting in the
repo's committed checkpoint JSONs / md5-pinned CRB, read back verbatim — not fresh statistics I
computed over the population).

## Verdict: **RED**

Two defects would make the registered read wrong (not merely worded loosely), and one would let an
unregistered outcome occur:

### RED-1 — `T_harn`/`SE_harn`/`Z_harn` are computed from the harness FULL score, not the matched-channel score revision 1 explicitly separated out

`REGISTRATION_DRAFT.md` §2.3/§2.4 (revision-1 item 2) is explicit and unambiguous: there are **two**
harness SEs, kept separate — `SE_full,harn` (the checkpointed FULL-score between-universe SE,
0.0063, **"INFORMATIONAL only, a design-power proxy"**) and `SE_harn` (the matched-channel
between-universe SE, **"the registered statistic's OWN SE... it, not (i), enters Z_harn"**). §2.4's
table defines `T_harn` as "mean over universes of **S_M,harn**" — the matched-channel score, the same
`s_M` quantity computed for production in §2.1.

`completion_residual_reads.py::compute_registered_statistics` never computes `s_M` for the harness at
all. It calls `reproduce_harness_byte_id()`, which globs `universe_seed*_S.json` checkpoints and pulls
`c["score_at_truth"]["no_bh"]["dark"]["mean"]` — the harness's own pre-aggregated **FULL** score (per
that function's own docstring: "the harness's own `_score_at_truth_by_class` output"). Both `t_harn`
and `se_harn` are built directly from this same `dark_full_score_means` list:

```
means = byte_id["dark_full_score_means"]          # FULL score, not S_M
t_harn = float(np.mean(means))
se_harn = float(np.std(means, ddof=1) / (len(means) ** 0.5))
z_harn = t_harn / se_harn
```

This is exactly `SE_full,harn`'s source and definition, relabelled as the registered `SE_harn`. The
script never reads the per-universe `event_likelihoods.csv` files that §2.3 names as Read B's data
(`tree2_20260830/b8_cal_harness_work_s4_postflip/seed9010NN_S/simulations/diagnostics/
event_likelihoods.csv`) — confirmed present on disk (96 such files under the harness root, headers
identical to production's: `event_idx, h, ..., B_num, D_tilde_phi, alpha_G_phi, ..., den_log_term,
num_log_term_no_bh, ...`; a per-universe `prepared_cramer_rao_bounds.csv` sidecar also exists, e.g.
`seed901012_S/simulations/prepared_cramer_rao_bounds.csv`) — so the matched-channel computation the
draft registers is fully computable from committed data, but the build node did not implement it.

**Consequence:** `Z_harn` as the script would emit it in real mode is the full-score Z, not the
matched-channel Z that both the disposition table and revision-1 item 2 require — i.e. the exact
conflation revision 1 was written to fix is still present in the executable. This is a
read-correctness defect, not wording.

### RED-2 — disposition table is not exhaustive; the script's own fallback proves it

`ρ` (§2.4) is registered as "evaluated only when |Z_prod| > 3" and the disposition table (§4) covers:
ILLEGITIMATE (`|Z_harn|>3` & `ρ≥0.5`), FLOOR-CONSISTENT (`|Z_harn|≤3` & `|Z_prod|≤3`), INTERMEDIATE(a)
(`|Z_harn|≤3` & `|Z_prod|>3`), INTERMEDIATE(b)/(c) (`|Z_harn|>3` & `ρ` in band, all requiring
`|Z_prod|>3` for ρ to exist), and NO-READ (gate failures). **No row covers `|Z_harn| > 3` AND
`|Z_prod| ≤ 3`** — harness-significant, production-not-significant. `SE_harn` (between-universe, 67
draws) and `SE_prod` (within-run per-event, N=1512) measure different things and are not nested, so
this combination is not excluded by construction. The build script itself hits this gap and improvises
an unregistered label:

```
else:
    disposition = "INTERMEDIATE (unclassified -- rho undefined, |Z_prod| <= 3)"
```

"unclassified" appears nowhere in §4's disposition table and carries no stage-5 action, no claim
writeback, and no fresh-RULE routing — a live violation of the design-gate requirement that "every
outcome returns as fresh RULE." If this branch is realised, the read produces an outcome the
registration never pre-registered a disposal for.

### AMBER items (wording / dead code, not read-correctness)

- **A1 — CRB md5 check computed but not enforced.** `run_dry_run` computes
  `production_crb_md5.match` and reports it, but `gates_green` (the printed pass/fail composite) does
  not include it, and `compute_registered_statistics` (real mode) is never even passed `--crb-md5` —
  the dataset-pinning STOP (CLAUDE.md 2026-08-20) is checked-but-not-gated in dry-run and entirely
  absent in real mode. Verified independently: the pinned md5 `9a1f2a14384a9281c97ca3be312ddaab` does
  match the file on disk today, so this is not currently wrong, but the script would not STOP on a
  future mismatch as the draft's own dataset-pinning convention requires.
- **A2 — stale `T0_MEAN_H_TOLERANCE = 1.0e-9` constant.** Revision-1 item 6 explicitly withdrew the
  literal 1e-9 test in favour of "round(computed, 6) == the 6-dp display anchor," which is exactly
  what the gating logic (`reproduces = round(mean_h, 6) == T0_MEAN_H_TARGET_IIIB_1D`) implements — the
  pass/fail path is correct. But the module constant is still named/valued `1.0e-9` and is echoed
  verbatim into the dry-run JSON's `"tolerance"` field, which will read as the withdrawn number to
  anyone consuming the record. Wording/reporting only; does not affect the gate outcome.
- **A3 — `check_gprecision` is defined but never called.** §5 lists g-precision among "Gates consumed"
  (with a "where available" qualifier consistent with it being informational), but no code path in
  either `run_dry_run` or `compute_registered_statistics` invokes it — it is dead code. Since the
  draft frames it as an availability-conditional cross-check rather than a hard STOP, this is a
  completeness gap in the build, not a wrong-read risk.

## Checks (data-existence / verbatim verification performed)

1. **Columns/keys exist** (headers/keys only, ≤5 rows loaded, no aggregate computed):
   - Production `event_likelihoods.csv` and the `joint_r1` replicate: identical 19-column headers,
     containing every column §2.1/§2.4 needs (`event_idx, h, alpha_G_phi, D_tilde_phi, B_num,
     den_log_term, num_log_term_no_bh, combined_no_bh`). GREEN.
   - `prepared_cramer_rao_bounds.csv` (seed61000): has `host_galaxy_index`, `in_catalog`; md5
     `9a1f2a14384a9281c97ca3be312ddaab` — matches the pinned `--crb-md5` in the launch block exactly.
     1590 data rows; the two JOIN-gate gap indices (1203, 1356) both have `host_galaxy_index == -1`
     (dark), so scored dark = 1514 − 2 = 1512, matching the script's `N_DARK_EXPECTED` and the draft's
     `N_Ḡ = 1512`. GREEN.
   - Harness checkpoints: 67 files matching `universe_seed*_S.json` under
     `b8_cal_harness_work_s4_postflip/`, matching `N_HARNESS_UNIVERSES`. Sampled one: has
     `universe.n_draw_requested`, `universe.seed`, `resolved_flags`,
     `score_at_truth.no_bh.{available,dark.mean,dark.sem}` — every key `reproduce_harness_byte_id`
     reads. GREEN.
   - Harness per-universe `event_likelihoods.csv` (needed for the matched-channel `S_M,harn` the draft
     registers but the script does not compute): 96 files exist under the harness root (e.g.
     `seed901012_S/simulations/diagnostics/event_likelihoods.csv`, 7135 rows, identical 19-column
     header to production), each with a `prepared_cramer_rao_bounds.csv` sidecar one directory up.
     Present and computable — see RED-1.
2. **SE/robust-SD formulas match code:** `SE_prod` matches exactly (`s_M.std(ddof=1)/sqrt(N)` on the
   production dark class — §2.2/§2.4). `SE_harn` does **not** match — see RED-1.
3. **Three-valued disposition + INTERMEDIATE sub-branches + fresh RULE:** table has the required
   structure and every named row returns a stage-5 fresh-RULE action, but the table is not exhaustive
   over the (Z_harn, Z_prod, ρ) state space and the script's own fallback proves it — see RED-2.
4. **Gates have bands + STOP consequences:** g-closure, g-population, g-znorm, g-byte-id, g-precision
   all have explicit thresholds and NO-READ is a named STOP disposition. g-precision is listed but its
   check function is unwired (AMBER-A3); the CRB md5 pin is checked but unwired into the gate composite
   and dropped entirely in real mode (AMBER-A1).
5. **Launch block has zero fresh choices; CLI matches script:** verified argument-by-argument — the
   12 flags in §7's command block (`--production-csv, --production-crb, --replicate-csv,
   --harness-root, --population, --h-lo, --h-hi, --h-true, --crb-md5, --catalogue-md5, --out,
   --dry-run`) match `build_parser()` exactly, same required/optional status. GREEN.
6. **Kill criterion verbatim + max_revisions=2:** fetched
   `graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md:45` directly — the q-completion-residual
   `kill_criterion` cell reads "registered arm fails to discriminate at its registered band after
   revision 2 -> park bounded-undetermined with the measured bound", matching §8b's quote
   character-for-character, and the line number (45) is correct. `max_revisions=2` is stated in §0 and
   sourced (line 146 of the same charter file, `r-completion-residual` row) to the same document.
   GREEN.
7. **Blindness-status line present:** §8b closes with an explicit "Blindness status:" paragraph
   (unblinded-artifact disclosure, band-freeze timing, executor/author blindness assertions). GREEN.

## Bottom line

Checks 1, 4 (partially), 5, 6, 7 pass. Checks 2 and 3 fail on read-correctness grounds: the built
scorer computes the harness statistic revision 1 designated "informational only" and reports it as the
registered `Z_harn`, and the disposition table has an unregistered gap that the script's own code
demonstrates is reachable. **Do not launch on this script build.** The fix is confined to
`completion_residual_reads.py` (implement `S_M,harn` per universe from the harness's own
`event_likelihoods.csv` + `prepared_cramer_rao_bounds.csv`, matching `compute_event_terms`'s production
path; wire the CRB md5 into the gate composite; either add the missing disposition row to §4 or prove
it unreachable and document why) — the registration draft's math and data plumbing are otherwise sound
and require no re-authoring.
