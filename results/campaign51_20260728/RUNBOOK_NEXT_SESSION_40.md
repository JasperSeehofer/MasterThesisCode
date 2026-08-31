# Runbook 40 — after the full-verification pass + the A18 flip landing (supersedes runbook 39)

**Read first.** Rows #278–#286 are ALL EXECUTED as of 2026-08-31. The author's row #278 ruling ("I approve all decisions and suggestions... please also do the full verification of both trees and your decisions via opus subagents and in parallel if possible") authorized and this session ran: PA-HIER-33 ratification, F-ii, the A18-conditional flip, A4-conditional, the B8.2 pilot, and a 42-opus-verifier full-verification pass of both trees. The pass came back clean enough that the A18 production arm ran, its verdict Z-CONFIRMED, and the `[PHYSICS]` production-default flip is now **committed** (`5e7fda16`). This runbook is the fresh-session entry point.

## 0. State of record (2026-08-31, all executed)

**Full verification (row #280).** 42 Opus verifiers + 2 top-tier adjudicators ran in parallel against both trees. **Tree 1: 19/19 CONFIRMED** (item 19 UNDETERMINED→CONFIRMED by independent mtime-span reconstruction; item 20 was deferred pending wave 3, which has since landed — see below). **Tree 2: 15/17 CONFIRMED, 2 REFUTED-DETAIL with headlines standing** (both certified [HIER] and 1D-rail chains reproduce end-to-end from raw data). Reports of record: `tree2_20260830/full_verification_20260831/FULL_VERIFICATION_TREE1_20260831.md`, `FULL_VERIFICATION_TREE2_DECISIONS_20260831.md` (+ `DEDUP_CONFLICTS.md`, `VERDICTS_ALL_README.md`). A process breach is disclosed and fixed: `verdicts_all.json` lost 41/42 records to a write race; verdicts were reconstructed from `work/` artifacts + re-execution, and per-verifier files are now the adopted fix.

**Wave 3 (row #279, #281, #283) — COMPLETE.** C0′ off-gate PASS, bit-identical both venues (job 6746274). Blind HEAD arrays (jobs 6746275/6746276, 84/84 tasks) completed; **A14 delta read is NOT MATERIAL**: raw Δmean_h were +0.002127 (iiib) / +0.003519 (joint_r1), both ≤ T_mat = 0.008. The item-20 end-verifier (Part 2, appended to `fanout1_20260829/END_VERIFIER_REPORT_PART1_20260830.md`) then found row #283's numbers used unit grid weights instead of the frozen T0 gradient-weighted scorer — **corrected deltas +0.002507 (iiib) / +0.004114 (joint_r1)**, still ≤ T_mat, PASS stands, and 1D is bit-identical at all 41 nodes (stronger than first reported). Verifier [DO] adopted and done: the T0 scorer is now frozen as an importable helper with a regression test.

**T2.2b (row #282) — EXECUTED, pure-input fork RESOLVED.** Runner-10 v3 ran arm (b) end-to-end in `tree2_20260830/t2_2b_arm_b_run/` (`T2_2B_RUN_RECORD.md`, `BAND_REDERIVATION_20260831.md`). Gates: BI PASS-AMENDED (cross-machine ulp floor), R/SCHEMA/ENG PASS. Registered readout: on-arm dark −0.0501 lands INSIDE the band [−0.097, −0.048] (ρ_eff 0.260) — the headline prediction CONFIRMS. Derived ARITH true-host transform BANKED: S_4D/S̄φ median **1.039** (66 hosts, h-stable). **Pure-input fork resolved: +157.92 binds; +123.11 is an O2 7-s.f. storage artifact** (catastrophic cancellation on 18 in-catalogue events — the discriminator was non-discriminating). Measured band for the A18 arm: 1D MAP ≈ 0.66 [0.65, 0.67], mean_h 0.652–0.673.

**A18 arm (row #285, #286) — SUBMITTED, VERDICT Z-CONFIRMED, FLIP EXECUTED.** Job 6747032 (55-node G-EXT grid) ran; verdict via the frozen T0 scorer: 1D map_h 0.665 / mean_h 0.66699 — inside both the registered [0.64, 0.72] band and the measured [0.65,0.67]/[0.652,0.673] band; floor mass collapsed 0.617→1.8e-4. **`[PHYSICS]` commit `5e7fda16`: `catalogue_leg_1d_mass_aware` production default flipped to `"auto"`** (auto engages "on" iff numerator+global-selection resolve "phi" and θ-divisor is "off"; silent "off" elsewhere; explicit "off"=COUNTERFACTUAL warned; explicit "on" logs `[PHYSICS] ACTIVE`). Sites touched: `bayesian_statistics.py`, `main.py`, `arguments.py` (choices auto/off/on), `validation/correspondence_1d.py`. Suite: 2006 passed / 6+1 skipped — the one nominal "failure" (T8 sky-selection margin) is NOT flip-related (see gotchas, §6). **The residual 1D rail (mean 0.667 vs truth 0.73, i.e. −0.063) is now OWNED: it is the mass-blind/mass-aware mismatch** — the dark-class completion-leg object, B8 [CAL]'s next centerpiece.

**A14 (row #280, #284, #286) — NOT MATERIAL, confirmed twice.** Corrected deltas +0.002507/+0.004114 both ≤ T_mat=0.008. The gradient-weighted T0 scorer correction is item-20 PART 2 in `fanout1_20260829/END_VERIFIER_REPORT_PART1_20260830.md`.

**Row #284 cross-check.** The author asked for a cross-check against an independently-prepared artifact ("Two Trees and the Residual Bias", `a8824799`) recommending the same decisions. Aligned and ratified: A18 flip authorization, the grid extension (G-EXT, already folded into the A18 submission), the A4 structural-default ratification (mz_sel/eff stands on the A14 PASS; the PROVISIONAL attribution cap on B7.1/B7.2 remains pending falsifier (ii)), the T5 78.9% retention retirement, and Arm R launch-when-cluster-allows. **Not covered: the Appendix-B scope word** — neither document recommends on it; it returns as the one remaining open word.

## 1. In flight at write time (2026-08-31, do not disturb without checking state first)

- **Runner-11**: 8-cell b-node pair under the T1.3 configuration, out-root `tree2_20260830/hier_s0_zwin_bnodes_run/` (subdirs seen: `s0a_seed900101`, `s0a_seed900102`, `s0a_seed900103`, `logs/`). This is the S0-B precondition — it dissolves the T1.2→T1.3 transfer assumption for the b-axis rather than asking the author to accept it. Read its state before touching the out-root.
- **Runner-9**: B8.2 S3 pilot, cell S, n_U=100, N=200 (the pilot proper, following the N=1588 ladder costing point referenced in row #284 item 5). Work root: `b8_cal_harness_work_ladder/`. Read via `--score-only` (prints bands, never a verdict) once a checkpoint exists; do not touch the work root while it runs.
- **Archive program**: `results/_archive/archive_run_wave2.sh` — archiving the wave-2 + wave-3 blocks. Check its log before assuming any wave-3/wave-2 out-root is still needed uncompressed.

## 2. Open author words

- **Appendix-B scope word** — the only remaining open item from row #284's cross-check. Neither the independent artifact nor this session's own recommendations state a position; it needs a fresh [RULE].
- **A4 — ratified-with-cap, PROVISIONAL** until falsifier (ii) runs. The falsifier (ii) is the class-G fleet rung (~40–60 CPU-h), the natural next cluster item alongside S0-B and the T5 k-scan (Arm S / Arm R).

## 3. Next-session queue, in order

1. **Read runner-11's output** (8-cell b-node pair, T1.3 config) — this is the S0-B precondition.
2. **S0-B submission** — PA-HIER-33 scorer implementation comes FIRST, before S0-B submits.
3. **Runner-9 pilot readout** (`--score-only`) → feed into F/coverage.
4. **Falsifier (ii)** — the class-G fleet rung (~40–60 CPU-h cluster item).
5. **T5 k-scan**, Arm S and Arm R.
6. **The residual −0.063** (mean 0.667 vs truth 0.73) — the dark-class completion-leg object. This is B8 [CAL]'s next centerpiece, not a bug to chase; it is now attributed to the mass-blind/mass-aware mismatch (§0 above).

## 4. Operating mode (author directive, 2026-08-31)

The orchestrator delegates ALL file writes and runs to subagents; the orchestrator's own role is limited to review, adjudication, and committing. Do not have the orchestrator itself write files or execute the registered measurement — that is what the subagent fan-out is for. (Verification pass, wave-3 submission, T2.2b, and the A18 arm were all built/run this way; the orchestrator's job was cross-checking decisive numbers and making the commit.)

## 5. Known machine gotchas from today (2026-08-31)

- **`pkill` self-match**: bracket patterns (`[p]ython foo`) are insufficient to avoid self-matching when the search path is present in your OWN command line (e.g. it appears in an argument you passed to `pkill`/`ps` itself). Kill by PID or PGID instead of pattern-matching.
- **`rsync -a` preserves symlinks**: when staging datasets, cluster copies that are themselves symlinks (e.g. into `injection_pool_mix200k_20260728`) will NOT be dereferenced by a plain `-a` sync. Use `-L` (dereference) plus an md5 manifest verified on both ends — this is exactly how the T2.2b injection-pool staging (707 files) was verified (707/707 after dereferencing).
- **Repo-root `simulations` symlink is REMOVED.** Its `/tmp/seed600_local` target is gone from this machine. Runner-10's symlink dance had transiently re-pointed the repo-root link at the seed61000 pool during T2.2b, which is why the T8 sky-selection test's one nominal "failure" during the A18 suite run was misattributed at first — T8's call path never reaches the mass-aware flag; it was only failing because of the transient repointing. **T8 now skips again** (link removed) until the `/tmp/seed600_local` pool is regenerated. Do not attempt to "fix" T8 by chasing the flip; regenerate the pool first.
- **Stale pre-rename `.pyc` caches** show phantom `MasterThesisCode` paths — cosmetic, but confusing if you're grepping for a path and it appears to still exist. Clear `__pycache__` if a path search returns something that isn't in the working tree.

## 6. Standing rules carried (do not re-learn)

Verifier output is evidence, not authority · subagents never run the registered measurement they built · never end a turn to wait on an untracked process · per-poll SSH, Monitor for watchers · every submission stamps its authorization · exoneration grep is for the MECHANISM, not the tag · row #223: production changes inside the tree are covered too, every gate still goes to the end verifier · a null offset derived for one estimator configuration does not transfer to another — pin every null to the arm's own likelihood structure · `--jobs>1` is dead in `hier_s0_driver.py`, always launch `--jobs 1` · SSH `ControlPersist` is 8 h and OTP-gated · `np.savez` silently appends `.npz` to a tmp path that doesn't already end in `.npz` — name tmp files ending `.npz` before the atomic replace.
