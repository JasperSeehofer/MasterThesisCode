# r-cone-loss — DESIGN GATE (computability-only), revision 1

Reviewer: fresh computability-only design-gate agent (no prior gate record opened; this
is an independent write-up — I did not open `DESIGN_GATE_stats.md`, `DESIGN_GATE_design.md`,
`DESIGN_GATE_provenance.md`, `INFORMATION_FORECAST.md`, or `cone_loss_result.json`, all of
which are either forbidden by task instruction or, in the case of the two extra
`DESIGN_GATE_*.md` files I discovered incidentally via a filename-only `grep -rl`, simply
never opened). Scope: `REGISTRATION_DRAFT.md` REVISION 1 (2026-09-03) and
`cone_loss_reads.py`, both in this directory. No aggregate (mean/sum/SD) was computed over
any registered population; no data file was read beyond its header/first few rows; the
scorer was never run (I read `cone_loss_work/cone_loss_gates.json` and
`cone_loss_work/cone_loss_result_rev1.json` from disk — both are the pre-existing
`--dry-run` build-node artifact, gates-and-census only, `dry_run: true`, no
Δh_cone/φ_cone/SE/Z field present in either — so this is evidence of the gate state, not a
statistic run by me).

## Check 1 — every column/file the statistic needs exists

GREEN.

- Production CRB `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`
  exists, md5 `9a1f2a14384a9281c97ca3be312ddaab` (matches G-1 pin exactly), 1591 lines
  (1590 data rows, matching draft's "1590 rows"). Header contains all of `CRB_COLS`
  (`qS, phiS, delta_qS_delta_qS, delta_phiS_delta_phiS, delta_phiS_delta_qS,
  host_galaxy_index, in_catalog`).
- Catalogue `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` exists, md5
  `c52c13b5cab61f6b3f04bbe202550969` (matches G-1 pin). Raw file has no header row (read
  via `GalaxyCatalogueHandler` with `names=` from `CatalogueColumns`); `THETA_S`/`PHI_S`
  are the handler's in-memory rename of the raw RA/Dec columns
  (`handler.py:539-541,171-180`), confirmed by reading the source, not assumed.
- Both venue run dirs (`run_20260902_graph1_headrebaseline_iiib`,
  `..._joint_r1`) have `GIT_COMMIT_AT_RUN.txt` = `1ec9514dd1808c48b18c0792dce558e5bba0f116`,
  matching the G-1 commit pin prefix `1ec9514d` in both.
- The production P6 log line exists verbatim as quoted in the draft:
  `run_20260902_graph1_headrebaseline_iiib/darksiren_emri_20260902_000633_h_0_73.log:8622`
  reads `P6 host-recovery (h=0.7300): 1D 66/76 hosts recovered/in-cat events seen
  (86.84211%)...` — exact match to draft §0 and §5 G-3.
- `simulations/diagnostics/event_likelihoods.csv` exists under both venue run dirs
  (65109 lines, i.e. 65108 rows, matching draft §6's "65,108 rows"), header includes
  `event_idx, h, combined_no_bh, combined_with_bh` (the columns the registered per-event
  score `s_e,c` needs), and the stencil h-values `0.725` and `0.735` are both present
  among the file's distinct `h` values (confirmed by listing distinct values, not by
  reading or aggregating the score columns).
- Both G-2 anchor-fleet CRBs exist on disk at the paths the draft/script expect
  (`p3_2d_fleet_20260825/bc_900121_work/seed900121/...` and
  `p3_b0_work/bc_900101_work/seed900101/...`), same `CRB_COLS` schema.
- Harness root `tree2_20260830/b8_cal_harness_work_s4_postflip` has exactly 67 `seed*_S`
  directories (matches draft's "67 post-flip S3 cell-S universes"); a sampled harness CRB
  (`seed901013_S`) and its `diagnostics/event_likelihoods.csv` both exist with the same
  schemas as production. A sampled raw checkpoint
  (`universe_seed901041_S.json`) contains the key path
  `score_at_truth.no_bh.catalogue_hosted` cited in draft §2 (existence-only check, no
  value read/reported).
- Source files the draft cites for reused conventions all exist:
  `prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (frozen T0 scorer),
  `fanout1_20260829/cmem_a1.py` (cone_radius/build_census precedent),
  `campaign51_20260728/realistic_20260729/p3_2d_fleet.py` (`_mahalanobis_check` style),
  `fanout1_20260829/b4_imp_stage1_forecast.py` (score convention).

## Check 2 — SE/robust-SD formulas fully specified and match the script

GREEN, with a structural note. The draft's formula chain is fully specified with every
symbol defined in the document: `SD_IN,c = 1.4826 · MAD_IN(s_e,c)` (robust, MAD-scaled,
per channel c, over the production IN class n_IN=66); `SE(Δh_cone,c) = SD_IN,c ·
sqrt(n_OUT + n_OUT²/n_IN) / I_c`; `Z = Δh/SE`; `M = T_mat/SE`. Nothing is left as a free
choice at read time.

`cone_loss_reads.py` does **not** implement any of this — real mode (no `--dry-run`)
raises `NotImplementedError` after the gate check, with an explicit comment naming the
verifier-independence contract ("a DIFFERENT agent implements and runs the real-mode
path"). This is not a gap relative to the draft: draft §7 registers exactly this
split ("builder runs ONLY `--dry-run`... a DIFFERENT agent runs the statistic"), and the
module docstring states the same contract independently. So there is currently nothing in
the built code that could contradict the formula — check 2 passes in the sense that the
spec is complete and unambiguous and the one piece of code that exists (the gate/census
half) does not implement or conflict with it. The runner-side implementation remains to be
written by a different agent and re-reviewed against this formula when it exists.

## Check 3 — bands three-valued incl. INTERMEDIATE(-UNPOWERED); every row returns as fresh RULE

GREEN, one wording note (AMBER-level, not RED). §4's disposition table has five rows —
IMMATERIAL-FLOOR-SHARE, CONE-OWNS-FLOOR, INTERMEDIATE-UNPOWERED, INTERMEDIATE,
INSTRUMENT/NO-READ — i.e. materially more than three-valued and explicitly includes both
the plain and the underpowered INTERMEDIATE variants. The M≥3 power gate is embedded as an
explicit `AND` clause inside the IMMATERIAL-FLOOR-SHARE, CONE-OWNS-FLOOR and INTERMEDIATE
triggers, and INTERMEDIATE-UNPOWERED's trigger is the M<3 complement stated
unconditionally on Δh/φ — the five rows are mutually exclusive and jointly exhaustive by
construction; no ordering ambiguity.

Wording note: the section header states "(every row returns as a fresh RULE)", but only
the CONE-OWNS-FLOOR and INTERMEDIATE/INTERMEDIATE-UNPOWERED action cells repeat "fresh
RULE" in the cell text itself; the IMMATERIAL-FLOOR-SHARE cell says "q-cone-loss SETTLED
(kill criterion...) — with the bound" and INSTRUMENT/NO-READ says "repair; no revision
consumed", neither literally re-stating "fresh RULE" in-cell. The section header already
covers all rows, so this reads as consistent, not contradictory, but a future reader
skimming only the action-cell text could miss that a SETTLED or an INSTRUMENT-DEFECT
outcome is still gated by an author RULE before it writes back. Recommend making it
explicit in-cell on revision 2, if there is one; not a computability defect as registered.

## Check 4 — gates have bands + STOP consequences

AMBER (one real gap; the four decisive gates themselves are GREEN/well-specified).

G-1 (three pins: catalogue md5, CRB md5, git commit) — explicit "STOP on mismatch."
G-2 (double anchor) — explicit "A miss = INSTRUMENT-DEFECT."
G-3 (join) — explicit "Mismatch ⇒ INSTRUMENT-DEFECT" (rev. 1 item 3).
G-4 (scatter law) — explicit "⇒ INSTRUMENT-DEFECT, STOP, fresh RULE," with the rev-1 fix
(binomial test against the nearest envelope edge, replacing the asymptotic band
comparison) correctly implemented in `cone_loss_reads.py` (`stats.binomtest(n_out_g4,
n_total_g4, p=nearest_edge, alternative="two-sided")`).

I read the existing `cone_loss_work/cone_loss_gates.json` (the build node's own
`--dry-run` output, not run by me) and it corroborates every specific number the draft
quotes: `n_out=10, n_in=66, n_in_catalogue=76` (13.2%); KS `D=0.06614822...,
p=0.87159...` (draft quotes 0.066/0.87 — matches); envelope binomial test against the
nearest edge `0.134` now returns `p=1.0`, `envelope_passed: true`, so **all four gates are
GREEN** in the on-disk record (`cone_loss_work/cone_loss_result_rev1.json` also confirms
`verdict: GATES-GREEN, dry_run: true`) — consistent with rev. 1 item 7's claim that the
one-clause fix was applied and re-run. Both G-2 anchors reproduce to tolerance
(`chord_ok`/`radius_ok`: true for both fleets).

The gap: §4's disposition table lists "G-1…G-4 red; **g-population red**" as the joint
trigger for INSTRUMENT/NO-READ, but no red/green criterion for `g-population` is ever
defined anywhere in the draft (§5's `g-population` line is a population-composition
disclosure only — seed-directory counts and the `--population 200` invariant, no
threshold), and `cone_loss_reads.py`'s `run_gates()` never folds
`g_population_disclosure` into `gates["passed"]` — it is computed and written to the
gates JSON but carries no `passed` field at all. So "g-population red" as a disposition
trigger is currently uncomputable / can never fire; it does not corrupt the primary
verdict (§2 already states the harness replicate this disclosure supports is "REPORTED to
d-calibration; not verdict-bearing here"), but the disposition table's own wording claims
a gate that the build does not implement. This is a spec-vs-code completeness gap, not a
defect that would make a read wrong — recommend either dropping the "g-population red"
clause from the INSTRUMENT/NO-READ trigger or defining what would make it red before
launch.

## Check 5 — launch block has zero fresh choices; script CLI matches

GREEN. Every flag in §7's launch invocation exists in `cone_loss_reads.py`'s argparse
(`--production-crb, --production-run, --replicate-run, --harness-root, --population,
--anchor-fleet-mker, --anchor-fleet-cmem, --sky-cone-k, --h-lo, --h-hi, --h-true,
--crb-md5, --catalogue-md5, --out, --dry-run`), and every value used traces to a value
already registered elsewhere in the draft body: the CRB/run/harness/anchor-fleet paths to
§2's population-of-record paragraph and §5 G-2; `--sky-cone-k 1.5` to the Invariants line;
`--h-lo/--h-hi/--h-true` to the stencil `(0.725, 0.735)` and `h_true=0.73` used throughout
§2/§3; both md5s to §2/§5 G-1. `--git-commit` is not passed explicitly at launch but its
argparse default (`1ec9514d`) equals the registered G-1 commit pin, so this is not an
unregistered choice either. No flag in the launch block is absent from the argparse, and
no argparse flag needed for the registered run is absent from the launch block.

## Check 6 — kill criterion verbatim quote + max_revisions=2

GREEN. I independently located `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` and confirmed line
46 reads (character-for-character): "measurement confirms the floor within its registered
uncertainty band -> settled as irreducible geometry; no fix pursued" — an exact match to
draft §4b's quoted text, including the `->` arrow and punctuation. `max_revisions 2` is
stated in the draft header ("max_revisions 2 (ORCHESTRATOR-DERIVED, charter §1.8/§1.13)")
and is consistent with the charter's own `r-cone-loss` row (line 157: "max_revisions 2
ORCHESTRATOR-DERIVED").

## Check 7 — blindness-status line present

GREEN. §4b carries an explicit "**Blindness status:**" line naming the specific leak
mechanism (a design-gate side effect, `DESIGN_GATE_stats.md`), stating band thresholds
were frozen before that record existed, that the registered read will be executed by an
agent that has not opened it, and separately disclosing (not conflating) that the
production OUT fraction (10/76) was seen by the registration author at stage 0 as
context, never as a verdict. This is consistent with what I independently observed: I
found `DESIGN_GATE_stats.md` exists (via a filename-only `grep -rl`, its contents never
opened) in this same directory, matching the draft's own disclosure.

## Overall

No RED (read-corrupting) defects found. One AMBER item of substance (check 4: the
"g-population red" disposition trigger is not operationally defined and is never computed
by the built script — currently a dead clause, not verdict-affecting since the harness
replicate it gates is explicitly non-verdict-bearing) and one AMBER wording note (check 3:
two of five disposition action-cells don't repeat "fresh RULE" in-cell, though the section
header already covers them). All four decisive gates (G-1..G-4) are correctly specified,
correctly implemented, and — per the existing on-disk `--dry-run` artifact I read but did
not generate — currently GREEN with numbers that match the draft's own citations exactly.
