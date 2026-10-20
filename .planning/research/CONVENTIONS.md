# Conventions ledger — EMRI → H₀ dark-siren inference

*Append-only convention lock (Layer-2 pack anatomy, vault:
`wiki/analyses/research-routine-packs-spec.md`). Every experiment/artifact declares which
conventions it uses; changing a CRITICAL row requires re-verifying all downstream consumers.
Never edit rows — append a superseding row and mark the old one superseded.*

> **Seeded 2026-09-21 by mission M65.** Every row below is a **transcription of a rule already
> binding in this repo**, with its source cited — M65 did not invent conventions. Rows marked
> `criticality: critical` are the ones whose change forces a downstream re-verification.

| # | convention | value | set by | criticality |
|---|---|---|---|---|
| C-1 | Fiducial cosmology | `OMEGA_M = 0.2726`, `H0 = 70.4 km/s/Mpc` — deliberately matched to the Barausse (2012) M1 EMRI-population cosmology (arXiv:1201.5888) for a self-consistent mock universe. The Planck-2018 mismatch is a **tracked systematic** (`docs/gates/G7_systematics_budget.md` row 6), not a bug | `constants.py`; CLAUDE.md "Known Bugs" G11 (GitHub #6 closed as design choice) | critical |
| C-2 | Injected/production H | `H = 0.73` | `constants.py` | critical |
| C-3 | Detection threshold | `SNR_THRESHOLD = 20` | `constants.py` | critical |
| C-4 | H₀ evaluation grid | `H_GRID_41` — 41 nodes spanning **0.600–0.860**; node 21 = 0.73. Edge nodes are 0.600 and 0.860 | `cluster/graph1_headrebaseline_iiib.sbatch`; `darksiren_emri/validation/correspondence_1d.py:353-358` | critical |
| C-5 | Fisher-matrix derivative scheme | 5-point stencil (`use_five_point_stencil=True`, default since Phase 10). Ref: Vallisneri (2008) arXiv:gr-qc/0703086 | CLAUDE.md, Known Bugs #4 [FIXED Phase 10] | critical |
| C-6 | PSD includes galactic confusion noise | `_confusion_noise()` in `LisaTdiConfiguration`. Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17) | CLAUDE.md, Known Bugs #5 [FIXED Phase 9] | critical |
| C-7 | Production H₀ pipeline | **Pipeline B only** (`bayesian_inference/bayesian_statistics.py`). Pipeline A removed in `c1571a2` (2026-05-01); per-source Cramér-Rao bounds from CSV, never a hardcoded distance error | CLAUDE.md Architecture | critical |
| C-8 | Detection probability | `SimulationDetectionProbability` — survival-estimator p_det built from the injection pool with `RegularGridInterpolator` look-ups (replaced the removed KDE-based `detection_probability.py`) | CLAUDE.md Architecture | critical |
| C-9 | Normalisation / p_det mode of record | `generator_marginal` + `--pdet_z_resolved` (adopted `ce6338e`, 2026-07-26), superseding `volume_deconv` as the named winner | vault `wiki/meta/hypothesis-ledger.md`, H-MTC-norm / H-MTC-genmarg | critical |
| C-10 | Physics-change protocol | **Hard gate.** No formula, constant, waveform parameter or model choice changes without derivation + dimensional analysis + limiting-case check + literature reference + regression test, presented *before* writing code | `.claude/skills/physics-change/`, `.claude/rules/physics-validation.md`; CLAUDE.md trigger table | critical |
| C-11 | Physics-gate evidence | Every gate run appends a row to `docs/gates/PHYSICS-GATE-LEDGER.md` (date · commit ref · step · verdict · target). Ledger starts 2026-07-30 and is **never back-filled**; a `[PHYSICS]` commit with no ledger row is a gate that cannot be shown to have run | CLAUDE.md "Math/Physics Validation Workflow" | critical |
| C-12 | Approval tagging | Every item put to the author carries **[DO]** (authorize work), **[RULE]** (ruling on evidence already in front of the author), or **[STANDING]** (pre-authorize a class). Binding default: *an approval never propagates to a decision whose inputs did not exist when it was given* — an uncomputed branch call, verdict or band comparison returns as a fresh [RULE]. "approved" for [DO], "ratified" for [RULE] | CLAUDE.md "Approval scope", codified 2026-08-14 | critical |
| C-13 | Decisions go in a reviewable artifact | Decision-gating proposals live in a persistent artifact (book chapter, docs page, standalone explainer) with the decision table inline — **never** a chat summary, which cannot be revisited, diffed or cited | CLAUDE.md "Proposing decisions", codified 2026-08-11 | high |
| C-14 | Pre-registration before running | Predictions are pre-registered **before** the run and never adjusted post-hoc. M65 evidence that this is honoured in practice: the m1 dossier's threshold block is byte-identical across all four dossier commits (md5 `c908790d…`), frozen ~11 h before the first array task started | `.claude/skills/research-cycle/`; vault `project-init` research contract; M65 `VERIFY-R1.md` §4 | critical |
| C-15 | Dataset pinning | Any multi-GB input not in version control (galaxy catalogues, cluster outputs) carries a **checksum pin at each consumer, STOP-gated on mismatch**. Machine-to-machine copies of "the same" file are not the same file | CLAUDE.md, codified 2026-08-20 after a stale local galaxy catalogue silently fed every local analysis | critical |
| C-16 | Run provenance | `--seed <int>` fixes the random state; when omitted a random seed is chosen, logged, and recorded with `git_commit`, `timestamp` and all CLI args in `run_metadata.json`. Cluster per-task seeding is `BASE_SEED + SLURM_ARRAY_TASK_ID`, owned by the `/cluster` skill | CLAUDE.md "Reproducible simulation runs" | critical |
| C-17 | Model / effort tiering | A top-tier session is an **orchestration** session. Mechanical stages → `sonnet`/`haiku`; derivation, adversarial verification, pre-registration authoring, physics interpretation → top tier. **At most ~3 top-tier agents per workflow**; every fanned-out stage runs `sonnet` regardless of adversarial label. `xhigh` effort only for adversarial verifiers, novel derivations, band/prereg authoring | CLAUDE.md "Orchestration", author mandates 2026-08-07 / 2026-08-14 | high |
| C-18 | Subagents never park | A subagent must never end a turn to "wait for a completion notification" on a process the harness does not track; every wait is a blocking foreground command | CLAUDE.md, codified 2026-08-20 after five parking incidents | high |
| C-19 | Cluster operations single source of truth | All bwUniCluster 3.0 submit/monitor/retrieve, SLURM flags, preflight and dataset provenance are owned by the `/cluster` skill; `cluster/preflight.sh` must return `VERDICT: READY ✓` before any submission. Preflight READY includes a **single-worktree** condition — clean up mission worktrees before submitting | CLAUDE.md; `.claude/skills/cluster/SKILL.md` | high |
| C-20 | Workspace lifetime | bwHPC workspace `emri` expires **2026-09-23**, **last extension used / 0 extensions remaining**; final results must be copied to persistent storage before then | `docs/campaign_redesign_51_design.md:216`; `results/prod2d_closure_20260818/CAMPAIGN_REPORT_20260819.md:89`; CLAUDE.md Constraints | critical |
| C-21 | Anchor vs truth are distinct references | A σ-distance must always name its reference. For the m1 read the registered anchors are the 0.73-run re-baseline means (**2D 0.6659, 1D 0.6670**, channel-specific), *not* the injected truth 0.67. Distances differ materially: 2.73σ from anchor vs **3.02σ from truth** (2D). Never write "2.73σ off" unqualified | M65 adjudication + `VERIFY-R1.md` §3 T2; `D_SEALED_REGISTER_DOSSIER.md` §2 | critical |
| C-22 | "Railed" is a threshold crossing, not pile-up | The registered rail test is edge-node mass `> 1e-3` on a 41-node grid (uniform ≈ 0.024/node), so it fires on any modest lower tail. For m1 the density is *declining* toward the edge (2D: 4.4 at h=0.600 vs 13.8 at h=0.610) with an **interior MAP (0.630)**. Report the number; qualify the word | M65 `VERIFY-R1.md` §3 T3 | high |
| C-23 | Channel roles | **2D `combined_with_bh` is the registered primary; 1D `combined_no_bh` is the replicate** ("2D primary; 1D must agree in cell") | `D_SEALED_REGISTER_DOSSIER.md` §2; `READ_RECORD_m1.md` §3 | critical |

## Open convention defects (raised by M65, not resolved here)

These are **not** conventions — they are places where the record currently lacks one, surfaced by
the m1 adjudication. Each needs an author [RULE]/[DO] before it can become a numbered row.

- **D-1 — no both-fire precedence rule.** The m1 cell table has no ordering for a posterior on
  which TUNED and NOT-TUNED-AT-RAIL both hold, and its INTERMEDIATE cell is registered as
  "neither", which on its face excludes "both". Docket R27 already names this. Will recur verbatim
  on (m2).
- **D-2 — the rail's consequence is registered two ways.** `REGISTRATION_DRAFT.md` §6/§8 makes
  `edge-node mass > 1e-3` a **NO-READ** ("nothing banked"); `D_SEALED_REGISTER_DOSSIER.md` §2
  reverses it to a banked **BOUND**. Opposite verdicts on the same posterior.
- **D-3 — no registered treatment for σ_measured < σ_anchor.** The dossier registered an
  expectation that σ_h would be ~9 % *wider* at N=1343; it came in 24 % (2D) / 20 % (1D)
  *narrower*, in the direction that keeps TUNED alive on ~9 % of margin.
- **D-4 — raw posterior data is not under version control.** `MANIFEST.md5` lists 85 entries;
  43 are tracked. The 41 posterior JSONs, `event_likelihoods.csv` and the scorer `t0_read.py` are
  absent, so every decisive number is prose-only testimony. Interacts with C-20.
