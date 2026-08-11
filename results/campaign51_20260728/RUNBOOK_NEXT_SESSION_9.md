# Runbook — next session(s) (written 2026-08-12, supersedes RUNBOOK_NEXT_SESSION_8)

RUNBOOK-8's queue is DONE: cross-term thread closed (NEGLECT-WITH-NUMBER ×4, ledger row 96),
M-2 residual dissolved (row 97), calibration gate built v1→v2 (row 98, KEEP-DIGGING DEFECT),
venue-transfer decider pre-registered + running. Two parallel session tracks from here:

- **Track A (this/main session): venue-transfer verdict.** Campaign array 6259842 (final 22
  chunks, 24 h limit) completes overnight 08-12/13. On completion: retrieve → collect/integrity
  → band-scored readout vs `results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md`
  (commit e77eecad; §11 has three appended deviation notes, all pending ratification) →
  independent adjudication → ship. Branches: TRANSFER-CONFIRMED (⇒ /physics-change intake on
  estimator photo-z handling, author-gated) / TRANSFER-REFUTED / MIXED / VENUE-CONFOUNDED.
  27/49 chunks already on cluster + retrieved-as-needed; T-0 anchor complete.
- **Track B (NEW session): HPC performance deep-dive on a branch** — author-mandated 2026-08-12
  (§2 below). Branch `perf/realistic-venue`; production-physics changes go through the FULL
  /physics-change gate on that branch, marked PENDING AUTHOR RATIFICATION.

## 0. State of the physics (post row 98)

- 2D displacement: carried by g_frac, ruled correct physics (R-A). Cross-term & M-2 residual:
  closed/dissolved with numbers (rows 96–97).
- 1D rail: starvation account (rail shape) + NEW co-candidate from the trustworthy v2 gate —
  σ_z-DOSED coverage collapse in the realistic ball venue (uniform +≈σ_z MAP bias, 0% coverage,
  delta-narrow posteriors; dose 0/+0.011/+0.035; B0 exactly on truth). Venue transfer to
  production = thread 17, the running decider.
- Paper #47: hold reason now "P–P leg FAILED — coverage DEFECT" (pending confirmation item vi).

## 1. Author ratification bundle (open, queued 2026-08-12)

1. Items (i)–(vi) of the 08-11 continuation record (ledger §5 "AUTHOR CONTINUATION"): deviation
   register D1–D8; DS-8 quotability; thread-17 co-candidate; venue-transfer as decider; DS-7
   open; #47 wording.
2. Venue-transfer §11 deviation notes ×3 (runtime blowout + resubmission; V-T5 sequencing;
   contention resubmission 6259842).
3. Perf roadmap physics items (§2): φ(M) interpolation swap + any GPU port = /physics-change.
4. Standing open queue: issue #53 (3σ window); N-2 adoption (queue item 3); ledger renumber
   item 6 (recommend §1b fork); LITERATURE_WARNINGS H-g status vocab; DS-5 F5 matched-population.

## 2. Track B mission: HPC performance deep-dive (author-mandated, autonomous, on a branch)

Author (2026-08-12, verbatim intent): implement everything on a branch; periodically re-check
"if the assumptions and approximations hold and if the errors are problematic as well as ...
the performance of the code and if the choices made still hold in the premise to accelerate
the pipeline." Strategic frame: realistic-venue campaigns become reusable infrastructure for
follow-on EMRI/LISA projects ([[realistic-venue-performance-goal]] memory).

Ground truth to start from (all committed):
- `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` — ranked plan + profiling data.
- Done: `--grain h` bit-identical mode (082d1e07, certified); grain archaeology + RNG analysis
  in workflow wf_b1b7a931-92a journal.
- Hotspot: 76% of seed time = h-independent φ(M) chain (`dark_mass_density_per_mass` →
  `R_eff_per_mbh` → mbh_mass_function/R0_per_mbh/duty_cycle_Gamma/kappa_cap) re-evaluated per
  call while cached table `_phi_dark_mass_log10_grid` (bayesian_statistics.py:1719) exists and
  is read only for its normalization scalar.

Mission spec:
1. Branch `perf/realistic-venue` off main. ALL work there; morning merges by author.
2. φ(M) interpolation swap = /physics-change (production files): full gate package — derivation
   of interpolation error bounds vs exact chain (incl. the kappa_cap kink at M=1e5 — the table
   must resolve it; check node placement), dimensional analysis, limiting cases, regression test
   pinning exact-vs-interp within stated tolerance, ledger row, [PHYSICS] commit on the branch,
   PENDING RATIFICATION marker. Counterfactual proof: venue-transfer smoke seeds byte-comparable
   (or within registered tolerance) exact vs interp.
3. GPU (cupy) port assessment→implementation for the same chain IF the interp swap alone leaves
   it dominant; xp-pattern; CPU/GPU parity tests; gpu-audit skill on changed files.
4. The author's periodic-audit habit: codify as a recurring "assumption & performance audit"
   ritual — proposal: research-cycle amendment (A6) or a scheduled routine — re-validate (a)
   approximation error budgets, (b) perf choices vs current unit economics, (c) prereg-assumption
   register entries. Draft the amendment text for author approval; do not self-adopt.
5. Node-topology packing rules → cluster skill gotchas (128-core cpu_il nodes; contention 1.7×;
   grain-vs-walltime rule; --test-only vs backfill EXP-61 logging).
6. Re-certify EVERYTHING touched (V-T5-style bit/numerical equivalence; adversarial verify).

## 3. Filler-task menu (author asked for suggestions; §3.1 is author-named)

3.1 **Repo rename** (beyond "master thesis") — phased, NOT a single yolo rename:
    a. GitHub repo rename (old URLs redirect) + local remote update + cluster remote update.
    b. Python package rename `master_thesis_code` → new name: large mechanical refactor
       (imports, pyproject, tests, sbatch, docs, book links, CI, Pages). Do on a branch with
       full-suite green; coordinate with cluster venv rebuild.
    c. OPERATIONAL TRAPS: local dir path is keyed into Claude memory/session state
       (~/.claude/projects/-home-jasper-Repositories-MasterThesisCode) and the garden registry
       Path column — renaming the local directory silently orphans memory + briefings; plan a
       migration step. Cluster ONE-repo rule: rename there in the same window. Workspace paths
       and DATA_INVENTORY references. Book/Pages URLs.
    d. Needs the author's name choice first.
3.2 Book ch14: the calibration-gate + venue-transfer arc (after Track A verdict).
3.3 GitHub hygiene: issues #4 (wCDM guard PR state), #53; milestone review vs rows 96–98.
3.4 Paper #47 verdict-independent sections: methods (pre-registration discipline, gate design),
    the dissolved-threads narrative (rows 96–97 are publishable regardless of the rail outcome).
3.5 W1 (rate-weights) / O2 (volume_deconv) reserved venue-transfer arms — only if Track A lands
    MIXED/needs them; seeds reserved (+46000/+47000), never post-hoc.
3.6 LITERATURE_WARNINGS H-g vocab conformance (5-min fix, author call on label).

## 4. Gotchas (new since RUNBOOK-8)

- Parallel GRAIN bounds task wall (W-PRE-19); margins vs CONTENDED timing (25-core tasks packed
  5/node run 1.6–1.9× slower); instruments write JSON only at end — timeouts leave NOTHING.
- `sbatch --test-only` ignores backfill: off by orders of magnitude for short wide jobs (EXP-61;
  log probe-vs-actual pairs each submission).
- dev_cpu_il QOS: MaxSubmit 4 / MaxRunning 1 / 30-min cap. cpu=192-core nodes, cpu_il=128.
- Authorization records: attribution-precise form (verbatim human words; derived itemizations
  marked as such; queued confirmations) — two classifier interventions taught this.
- Silence ≠ death for background agents (W-CONF-46): check liveness, don't infer from quiet.
- Cluster repo symlink bridges: results/ inputs symlinked to $WS (CRB CSV, injection pool) —
  md5-verify against prereg pins after any re-stage.
- scontrol walltime extension: denied for users; size right or resubmit.

## 5. Provenance quick map (this arc)

| artifact | where | commit |
|---|---|---|
| calgate v1 (GATE-NOT-TRUSTWORTHY) | results/calibration_gate_20260808/ | 3a572897 |
| calgate v2 (KEEP-DIGGING(b)) | results/calibration_gate_v2_20260810/ | 64abd5f6 (+065e7f58 prereg) |
| ledger rows 96–98 + rulings | gate_b_20260730/BIAS_HISTORY_LEDGER.md | a9c2cae0…689cba49 |
| venue-transfer prereg+instrument | results/venue_transfer_20260811/ | e77eecad + 2ece8801 |
| VT campaigns | arrays 6252702 / 6253922 / 6259842 (cluster) | outputs partial→27/49, final wave running |
| perf: h-grain + roadmap | validation/venue_transfer.py + perf/ | 082d1e07 + 2f2f5975 |
| book ch12/ch13 | book/site/ | 28aa21d3 + 2c6fdbc7 |

Cluster workspace expires 2026-09-23. GLADE reduced catalogue is gitignored (~1.6 GB) — stage
from dev box if a fresh cluster clone ever happens (see cluster skill gotcha 2).
