# Handoff — Weekend deep code review (2026-07-04 → 07-06)

Fresh-session entry point. The Phase-2 campaign runs autonomously on the cluster all
weekend (2FA re-auth impossible before Monday); this session uses the time for an
in-depth, Workflow-orchestrated review of the whole codebase. User-authorized scope:
improve the local code base and push; worst case we revert; verify with local test
runs only. The user explicitly requested "a very detailed code review with workflows"
— that is the standing multi-agent opt-in for this work.

## 0. Standing state (do NOT rediscover; verify only if something looks off)

- Branch `physics/campaign-depth-pv` @ `8c59cce`, PR #22 open, tag
  `campaign-phase2-base` = `b6bf57d`. Campaign design + away-plan: memory
  `project_campaign_phase2.md` + runbook `.planning/CAMPAIGN-PREP-PHASE2.md` §4b/§4c.
- **Cluster is UNREACHABLE until Monday** (2FA ControlMaster expired; `ssh bwunicluster`
  fails with "Permission denied (publickey,keyboard-interactive)" — expected, do not
  retry in loops). The campaign self-drives: login-node orchestrator submits seeds
  2000–6000; local `results/campaign_phase2_runs/watch_and_retrieve.sh` (detached,
  survives session clears) resumes telemetry automatically after the user's next
  interactive 2FA login. Cluster repo pinned at `b233375` (+ $HOME Lustre EIO blocks
  git pulls there — repair + re-sync is a MONDAY task, runbook §4c).
- **§7b PV test**: 26 local evaluations DONE incl. the h ≤ 0.785 grid extension
  (detached queue; check `results/pv_correction_test_20260703/run_queue.log` says
  "ALL RUNS DONE"). FIRST TASK of the new session (§2 below): finalize the analysis
  and post to issue #16. Preliminary (9-value grid): 1D Δmean(live−noPV) ≈ −0.0137
  (≈2.7 posterior widths — seed600 is all-low-z, the designed upper bound); 2D was
  edge-railed at 0.765, hence the extension.
- Detached local processes that must keep running: the retrieval watcher and (until
  done) the PV queue. `pkill` patterns must use the `[x]` bracket idiom — a plain
  `pkill -f name` matches its own wrapper shell (bit us twice on 2026-07-03).

## 1. Weekend ground rules

1. **Campaign-safety classification for every proposed fix** (hard rule):
   - **sim-semantic** (would change what the GPU campaign is computing now, e.g.
     population draws, waveforms, SNR, Fisher): PROPOSE ONLY — file an issue, never
     land; re-simulation is a user decision.
   - **inference-semantic** (changes posterior numbers: kernels, integrals, p_det
     consumption, normalizations): allowed ONLY through `/physics-change` with pin
     updates in the same diff, AND flag "re-evaluate tier vs `campaign-phase2-base`"
     in DATA_INVENTORY. Default: DEFER to post-harvest unless the bug is severe.
   - **neutral** (refactors, tests, guards, docs, dead-code removal, typing, plotting):
     free to land with atomic commits.
2. Full fast suite green (`uv run pytest -m "not gpu and not slow"`) + `mypy` clean
   before every commit; pre-commit reformat aborts the first commit attempt — re-add
   and re-commit (recurring today).
3. Work on a NEW branch `review/codebase-20260704` stacked on
   `physics/campaign-depth-pv` (keeps PR #22 reviewable); separate PR at the end.
4. Never touch `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`,
   `m_th_map_nside32.npy` (C1 frozen), or anything under `simulations/`.
5. GSD/GPD routing per CLAUDE.md: physics-semantic work → `/physics-change` gate;
   the review itself is software work.

## 2. FIRST: finalize §7b (30 min, before the review)

1. Confirm `run_queue.log` ends with ALL RUNS DONE (both arms have 13 posteriors).
2. Re-run combine per arm (from each arm's `cwd/`, working dir `../simulations`,
   `--combine --allow_low_pdet_coverage`), recompute MAP/mean/std per channel on the
   13-value grid; verify the 2D posteriors are no longer edge-railed (if still railed
   at 0.785, extend again by 4 values — the queue script is idempotent).
3. Refine to the event INTERSECTION: the arms differ by one event (3342 vs 3343 used).
   Identify it from the per-h JSONs and recompute both combined posteriors on the
   shared set (production combine machinery; per-event exclusion is supported by the
   JSON structure — inspect, don't guess).
4. Write `results/pv_correction_test_20260703/ANALYSIS.md` (numbers, sign convention
   impact = live − noPV, upper-bound framing: seed600 all-low-z where all 709k
   corrections live; campaign events at depth 1.5 far less sensitive) + commit +
   post the result to issue #16 (stays open only if a re-scope is warranted;
   otherwise close it citing the marginalization commit `8568d9f` + this bound).

## 3. THE CODE REVIEW (Workflow-orchestrated, the session's main work)

Target: every `.py` under `master_thesis_code/` (~40 files), the test suite, and
`cluster/*.sh|*.sbatch`. Roughly: fan out dimension reviewers → dedup → adversarial
verify → fix-wave (neutral fixes only, per §1) → report + issues.

### Phase A — dimension fan-out (schema'd findings)

One agent per dimension; each returns findings with: id, file:line, severity
(critical/major/minor/nit), campaign_safety (sim-semantic/inference-semantic/neutral),
evidence (what the code actually does — quote it), proposed_fix, effort. Point each
agent at the SPECIFIC suspicions below — they are informed leads from today's recon,
not exhaustive lists:

1. **likelihood-core** (`bayesian_inference/bayesian_statistics.py`, ~2300 lines):
   single_host_likelihood vs Gray et al. (2020) A.9/A.10/A.19; fixed_quad n=50/100
   adequacy now that windows widened (σ_v) and z_max(h) deepens with the new pool —
   check node coverage over the completeness feature (f-structure lives z<0.2 while
   D(h) integrates to z_max(h)≈3+ post-dt²; sweep flagged quadrature dilution);
   narrow-σ_z D_g≈P_det(z_g) approximation (docstring ~line 560) degrades as σ_eff
   grows — quantify; MC importance sampling (proposal=prior ⇒ weights=p_det) after
   the G2d Eddington-M shift — is proposal still = prior exactly?; partition-norm
   ratio-of-sums; worker-global lifecycle (init_worker vs test-set globals).
2. **selection-function** (`bayesian_inference/simulation_detection_probability.py`):
   survival estimator correctness (searchsorted conventions, suffix weights, sky
   bands, bandwidth_scale), h-invariance claim end-to-end, the new gates (edge cases:
   empty pool, all-NaN columns), M_z clamp interaction with the 0.9/1.1 axis padding.
3. **catalogue-handler** (`galaxy_catalogue/handler.py`): parse chunking (append-mode
   hazard), prune correctness, BallTree query geometry (ecliptic rotation COORD-03),
   the ±1σ candidate z-window vs the σ_v-widened kernel (documented exclusion —
   quantify the candidate-list loss at spec-z hosts), HostGalaxy field provenance,
   **memory at depth 1.5**: the in-RAM pruned catalogue grew ~4–5× (z≤1.5 keeps far
   more of the 22.6M rows) — measure RSS on load + BallTree build, times 14 workers
   fork-COW on the 16-cpu eval nodes; flag if the 6h/16cpu eval jobs risk OOM
   (SLURM default mem per cpu!) — this is the one finding that could matter for the
   RUNNING campaign; if real, the Monday fix is an sbatch `--mem` line.
4. **completeness** (`galaxy_catalogue/pixel_completeness.py` + its consumers):
   Schechter/gammaincc usage, C1 coupling sites, W_k weights, the L_cat-vs-completion
   double-count tension for real z>0.5 galaxies with f≈0 (adjudicate as a Paper-B
   design note + issue, NOT a code fix).
5. **simulation-loop** (`main.py`): the exception-`continue` catalogue in both loops —
   which swallowed classes could bias selection (e.g. RuntimeError on loud/edge
   waveforms) and should at least be counted per class; signal.alarm nesting/reset on
   ALL paths; injection flush interval vs wall-cap data-loss window; provenance
   completeness (which stages still lack git_commit).
6. **physics-relations & cosmology** (`physical_relations.py`, `cosmological_model.py`,
   `datamodels/parameter_space.py`): adjudicate known bugs #6 (wCDM silently ignored —
   propose fix or explicit NotImplementedError), #8 (WMAP fiducial — G11 says
   deliberate, reconcile the CLAUDE.md entry), #9 (galaxy.py (1+z)^3 — orphaned,
   fold into dead-code decision); dist_to_redshift fsolve bracketing at 13 Gpc;
   derivative_epsilon choices vs five-point stencil.
7. **hpc-gpu compliance** (`parameter_estimation/`, `LISA_configuration.py`,
   `memory_management.py`, `decorators.py`): **fix known code-health bug #1 this
   weekend** (unconditional `import cupy` in LISA_configuration.py — guarded-import
   pattern per `.claude/rules/hpc-gpu.md`; it's on the "fix when touched" list and
   it's neutral); xp-pattern violations; hot-path transfers; USE_GPU threading.
8. **tests-quality** (`master_thesis_code_test/`): inventory numeric pins vs
   structural tests; uncovered critical paths (handler parse_to_reduced_catalog is
   UNTESTED end-to-end; ball-tree candidate selection; posterior_combination
   strategies; arguments.py flag plumbing); fixture realism vs depth 1.5; marker
   hygiene; the 9 data-gated skips (should assert skip-reason messages).
9. **dead-code & drift**: Pipeline A remnants (datamodels/galaxy.py GalaxyCatalog,
   `bayesian_inference/bayesian_inference.py` shim + known bug #7,
   LUMINOSITY_DISTANCE_THRESHOLD_GPC, `single_host_likelihood_grid` stub,
   `parse_to_reduced_catalog_with_reduced_errors`, `scripts/quick_snr_calibration.py`
   now moot post-gate-disable, detection.py dead scatter clips A4) — propose ONE
   coherent deletion PR section with test retargeting; stale comments/line-refs;
   CLAUDE.md Known Bugs section reconciliation (several entries outdated today).
10. **repro & provenance**: rng threading completeness (every stochastic site reaches
    --seed?), run_metadata coverage per stage, determinism of evaluate under
    num_workers variation (worker scheduling vs seeded MC — verify the G4 claim).
11. **plotting & callbacks** (`plotting/`, `callbacks.py`): `apply_style()` compliance
    (user feedback memory), depth-1.5 axis ranges/defaults, figure-data provenance
    (W-TOOL-11: green tests ≠ correct figure), HORIZON template consistency.
12. **cluster-ops bash** (`cluster/`): shellcheck-grade pass over ALL scripts incl.
    today's edits (private-CWD blocks, orchestrator, resubmit_failed rewrite) — today
    they got `bash -n` only; check quoting, set -u interactions, empty-glob behavior,
    ssh BatchMode assumptions.

### Phase B — dedup + adversarial verification

Barrier-collect all findings; dedup by (file, line-range, topic). Then verify:
critical/major findings get an adversarial refuter (2 independent refuters for
anything inference- or sim-semantic; kill on majority-refute). Minor/nit pass
through unverified but marked.

### Phase C — fix wave (neutral only) + report

- Sequential fixes grouped by file-area to avoid conflicts (or worktree isolation if
  parallelizing); each fix: atomic commit, tests for the fixed behavior where
  meaningful, full fast suite + mypy before each commit.
- Semantic findings: GitHub issues (labels `physics`/`bug`/`design-choice` +
  `paper-blocker` where apt), referenced in the report.
- Artifact: `docs/reviews/CODE-REVIEW-20260704.md` — verified findings table
  (incl. refuted-with-reason), fixes landed (commit refs), issues opened, and a
  short "campaign risk assessment" section (anything that could affect the running
  campaign — the eval-node OOM check from dimension 3 goes here first).
- Update CLAUDE.md Known Bugs to match reality; `/pre-commit-docs` before the final
  commit; push branch + open PR (base: `physics/campaign-depth-pv`).

### Sizing guidance

Ultracode-scale: 12 reviewers + ~2×verifiers on majors + fixers. Reviewers need
`effort: 'high'` on dimensions 1–3 (dense physics code); mechanical dimensions
(9, 11, 12) can run lower. Expect 2–3 workflow rounds: review+verify, then fix,
then a completeness-critic pass ("which files did NO reviewer actually read?").

## 4. Monday (unchanged, runbook §4c)

Interactive 2FA login → watcher auto-resumes → harvest campaign per §4b criterion →
repair cluster git sync ($HOME EIO) → merge PR #22 (+ review PR) → re-sync cluster.
