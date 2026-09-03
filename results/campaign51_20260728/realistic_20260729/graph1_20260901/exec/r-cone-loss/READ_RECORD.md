# r-cone-loss — READ RECORD (disjoint reader, real-mode launch execution)

Role: DISJOINT READER for node `m-cone-loss`, launched under docket 2.2 after the
GREEN computability gate
(`DESIGN_GATE_rev1_computability.md`, this directory). This record is **VERDICT-FREE**:
no ruling, no promotion, no recommendation. It reports exactly what the launch-block
execution produced.

Files not opened by this agent, per task instruction: `DESIGN_GATE_stats.md`,
`INFORMATION_FORECAST.md`, `cone_loss_result.json` (the pre-existing top-level file in
this directory — its filename listing was seen via `ls`, its contents were never read).
No production pipeline, cluster command, or edit under `darksiren_emri/` was run.

## 1. Exact command executed (from `REGISTRATION_DRAFT.md` §7, REVISION 1; real mode,
i.e. `--dry-run` omitted, run from repo root)

`--out` deviation (per task instruction): §7's launch block names
`cone_loss_result.json`; since that name is on the forbidden-to-open list (it is the
superseded pre-existing artifact per REVISION 1 item 7), this run used
`cone_loss_result_rev1_read.json` instead.

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --production-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib \
  --replicate-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1 \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip --population 200 \
  --anchor-fleet-mker results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825 \
  --anchor-fleet-cmem results/campaign51_20260728/realistic_20260729/p3_b0_work \
  --sky-cone-k 1.5 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_result_rev1_read.json
```

Run once, unmodified script (`cone_loss_reads.py`, on-disk revision timestamped
2026-09-03 20:56, git-modified-in-worktree relative to HEAD `25be7a66` — not touched by
this agent).

## 2. Exit code

**1** (unhandled Python exception; no `SystemExit` — a `NotImplementedError` propagated
out of `main()`).

## 3. What the run actually did (headline fact, stated before the gate table because it
governs everything below)

`cone_loss_reads.py`'s real-mode path (no `--dry-run`) runs gates G-1…G-4 and the
production census exactly as `--dry-run` does, but — by the script's own module
docstring and an explicit `raise NotImplementedError(...)` at the end of `main()` — it
**does not implement the registered §2 statistic** (Δh_cone, φ_cone, SE, Z, the
leave-out cross-check, or the harness Δs replicate) in either mode. The comment in the
script (lines 569–575) and the module docstring (lines 7–13) both state this is
intentional: "a DIFFERENT agent implements and runs the real-mode path." No such
implementation exists anywhere in this file. Because all four gates passed, the script
never reached even its own gate-fail write branch — it went straight from printing the
census to raising `NotImplementedError`, so **no `--out` file was written** (confirmed:
`cone_loss_result_rev1_read.json` does not exist on disk after the run). The registered
intermediates in §5 and the band-clause table in §6 are reported as NOT COMPUTED /
INPUTS DO NOT EXIST for this reason — this is not a partial read, it is a read that
terminated before the scored statistic existed to be read.

Side effect disclosed: running the script (as instructed, unmodified) regenerated
`cone_loss_work/cone_loss_gates.json` and `cone_loss_work/cone_loss_census.csv` in
place (both already showed as modified-in-worktree before this run per `git status`,
i.e. this is the script's normal, expected side effect of executing its gate/census
path again — not a change to the script itself, and not a change made by editing any
file).

## 4. Gate results (as the script reports them; `passed: true/false` mapped to
GREEN/RED; STOP is the script's own declared consequence, taken from
`REGISTRATION_DRAFT.md` §5, not asserted independently)

| gate | sub-check | script result | GREEN/RED | STOP declared? |
|---|---|---|---|---|
| G-1 catalogue pin | md5 `c52c13b5cab61f6b3f04bbe202550969` | matches | GREEN | n/a (matched) |
| G-1 CRB pin | md5 `9a1f2a14384a9281c97ca3be312ddaab` | matches | GREEN | n/a (matched) |
| G-1 git-commit pin | production + replicate `GIT_COMMIT_AT_RUN.txt` both `1ec9514dd1808c48b18c0792dce558e5bba0f116`, prefix `1ec9514d` | both match | GREEN | n/a (matched) |
| G-2 anchor (MKER-6) | `bc_900121_work` seed 900121 event 20: chord `0.001674659860716462` vs expected `0.00167466` (tol 5e-10); radius `0.0014956979545757095` vs expected exact | both within tol | GREEN | n/a |
| G-2 anchor (CMEM-A1) | `bc_900101_work` seed 900101 event 0: chord `0.01166569410071811` vs expected `0.0116656941007181`; radius `0.035912194615445196` vs expected `0.0359121946154451` | both within tol | GREEN | n/a |
| G-3 join | `n_total_crb_rows=1590`, `scored_set_size=1588`, `n_in_catalogue=76`, P6 numerator (66) = n_IN (66) | matches | GREEN | n/a |
| G-4 KS clause (decisive) | Mahalanobis² vs χ²₂, `D=0.06614822414302035`, `p=0.8715984091477792` (α=0.05) | p ≥ α | GREEN | n/a |
| G-4 envelope clause (rev.1 binomial form) | n_out=10, n_total=76, nearest edge 0.134, two-sided binomial p=1.0 (α=0.05) | p ≥ α | GREEN | n/a |
| g-population disclosure | `n_seed_S=67`, `n_seed_T=25` under harness root, `--population 200` | disclosed, not a pass/fail gate in the script (no `passed` field — matches the DESIGN_GATE_rev1_computability.md check-4 finding that this clause is uncomputable as currently built) | N/A — not computed by the script | n/a |
| **Overall `gates["passed"]`** | all of G-1(×3)/G-2/G-3/G-4 | `true` | **GREEN** | run proceeded past the gate-fail branch (which is why it reached `NotImplementedError` instead of writing an `INSTRUMENT-DEFECT` `--out` file) |

## 5. Closure residual

Not computed. The script contains no code path that computes a "closure residual" (the
term does not appear in `cone_loss_reads.py`); real mode terminates at the
`NotImplementedError` before any such quantity could be derived.

## 6. Every registered intermediate (§2: per-class n, means, robust SD, SE, Z, ρ/φ)

| intermediate | §2 registered definition | value from this run | source |
|---|---|---|---|
| n_in_catalogue | in-catalogue events (`in_catalog` true, `host_galaxy_index ≥ 0`) | **76** | this run's stdout `CENSUS:` line + `g3_join.n_in_catalogue` |
| n_OUT | in-catalogue events whose true host is outside the 1D sky cone | **10** | this run's `g4_scatter_law.n_out` / `g3_join.n_out` |
| n_IN | in-catalogue events whose true host is inside the cone (the paired comparand class) | **66** | this run's `g3_join.n_in` (matches P6 log numerator 66/76) |
| n_DARK | scored_set_size − n_in_catalogue (draft §2: "DARK (1512)") | **1512** (1588 − 76, arithmetic on this run's own reported `scored_set_size=1588` and `n_in_catalogue=76`; not a field the script emits directly) | derived from `g3_join.scored_set_size` and `g3_join.n_in_catalogue` |
| s̄_IN,1D, s̄_IN,2D (mean per-event score, IN class) | §2 | **NOT COMPUTED** | no code path reached |
| s̄_OUT,1D, s̄_OUT,2D | §2 | **NOT COMPUTED** | no code path reached |
| SD_IN,c (robust, MAD-scaled, rev. 1) | §2 | **NOT COMPUTED** | no code path reached |
| SD_IN,c (plain sample SD, disclosed alongside) | §2 rev. 1 item 1 | **NOT COMPUTED** | no code path reached |
| 2-outlier sensitivity (largest 2 \|s_e − median\| IN events) | §2 rev. 1 item 1 | **NOT COMPUTED** | no code path reached |
| Δh_cone,1D, Δh_cone,2D | §2 | **NOT COMPUTED** | no code path reached |
| SE(Δh_cone,1D), SE(Δh_cone,2D) | §2 rev. 1 formula | **NOT COMPUTED** | no code path reached |
| Z,1D / Z,2D | §2 | **NOT COMPUTED** | no code path reached |
| φ_cone,1D, φ_cone,2D | §2 | **NOT COMPUTED** | no code path reached |
| M (materiality margin, §3 rev. 1) | §3 | **NOT COMPUTED** | no code path reached |
| leave-out (T0 scorer) Δmean_h | §2 cross-check | **NOT COMPUTED** | no code path reached |
| harness replicate f_OUT,harn, Δs (§2) | §2 | **NOT COMPUTED** | no code path reached |

## 7. Three-valued outcome of each band clause (§4 disposition table, §3 bands) —
existence contract for inputs; VERDICT-FREE

Every clause below requires at least one of {Δh_cone,1D, φ_cone,1D, Z, M, SE} (§6). None
of those five quantities exist as an output of this run (§6 shows all NOT COMPUTED). Per
the existence contract, every clause is therefore reported as **INPUTS-DO-NOT-EXIST**,
not as TRUE, FALSE, or a computed band label — reporting anything else would fabricate a
value the run never produced.

| disposition row (§4) | trigger clause | outcome |
|---|---|---|
| IMMATERIAL-FLOOR-SHARE | \|Δh_cone\| < 0.008 AND φ < 0.2 AND M ≥ 3 | INPUTS-DO-NOT-EXIST (Δh_cone, φ, M not computed) |
| CONE-OWNS-FLOOR | \|Z\| > 3 AND φ ≥ 0.5 AND M ≥ 3 | INPUTS-DO-NOT-EXIST (Z, φ, M not computed) |
| INTERMEDIATE-UNPOWERED | SE(Δh_cone,1D) > T_mat/3 (M < 3) | INPUTS-DO-NOT-EXIST (SE, M not computed) |
| INTERMEDIATE | M ≥ 3 AND (\|Z\|>3 AND 0.2≤φ<0.5; or \|Δh\|≥0.008 with φ<0.2; or 1D/2D disagree; or linear vs leave-out disagree >2·SE) | INPUTS-DO-NOT-EXIST (all named quantities not computed) |
| INSTRUMENT / NO-READ | G-1…G-4 red; g-population red | **CLAUSE FALSE for the computed half**: G-1…G-4 are all GREEN this run (§4 above) — the trigger's G-1…G-4 conjunct does not fire. The `g-population red` conjunct is, per `DESIGN_GATE_rev1_computability.md` check 4 (confirmed independently by inspecting `run_gates()` above), never computed by the script — `g_population_disclosure` carries no `passed` field — so that conjunct is itself INPUTS-DO-NOT-EXIST. Net: this row's trigger cannot be affirmed by anything in this run, but it also cannot be evaluated to fully FALSE because one of its two conjuncts is uncomputable as built. |

Separately, and not a band clause: the run's own terminal state (`NotImplementedError`,
exit code 1, no `--out` file written) is itself outside the disposition table's five
rows entirely — the table has no row for "the statistic was never computed by the
script." That is a fact about the build, reported here, not a sixth disposition.

## 8. Files touched / produced by this run

- `cone_loss_work/cone_loss_gates.json` — regenerated (script's normal side effect;
  content matches §4 above).
- `cone_loss_work/cone_loss_census.csv` — regenerated (76 rows, in-catalogue census).
- `cone_loss_result_rev1_read.json` (the renamed `--out` target) — **NOT created**; the
  script never reached its write statement in real mode with gates passing.
- `cone_loss_reads.py` — not modified by this agent.
- Not opened: `DESIGN_GATE_stats.md`, `INFORMATION_FORECAST.md`, `cone_loss_result.json`
  (top-level, pre-existing, superseded per REVISION 1 item 7).
