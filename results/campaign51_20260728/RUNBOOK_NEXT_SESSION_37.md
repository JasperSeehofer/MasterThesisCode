# Runbook 37 — the fan-out session (supersedes runbook 36; [FABLE-ORCH 2026-08-29])

**Read first.** This session is the FRESH context for the first large-scale fan-out of the
residual-bias leads, chartered on 2026-08-29. The design lives in the **Fan-out Charter**
artifact (`claude.ai/code/artifact/500fef3e-e235-4873-a415-7bb0f9313d8e`) and is
mirrored in §2 below; the author's root goal of record (row #221, verbatim):

> "a scientifically correct mathematical setup for the bayesian inference in 1d and 2d which
> should be unbiased up to the level where we have to admit that the information has starved"

## 0. State at hand-off (2026-08-29)

- **Grants in force:** row #216 ("all approved"), row #220 ("ratified"), **row #221** (the
  entire Arc Follow-ups document approved as recommended: S0-A build+run · S0-B costing ·
  F-ii = REDESIGN (log-symmetric, k = 3, ε = 0.27 %, adopt only after a registered
  counterfactual) · [CMEM] A1 → conditional A2 · θ-hook `s`-placement alignment gate ·
  re-open population misspecification · register an impostor-drag attack).
- **Not yet ratified:** the charter itself (branches, waves, amendments F1–F5) and the
  wave-1 Workflow launch. Open the session by presenting the charter's three asks; do NOT
  launch until the author rules.
- **Built and green:** θ-hook C1+C2 (`d40fe5c8`), WGEOM instrument + ε-table banked,
  CMEM reads. Suite 1846 passed at last run.
- **Cluster:** workspace `emri` expires **2026-09-23, 0 extensions**. Archival Option A
  (`results/_archive/archive_run_20260828.sh`) was running at hand-off — check its log;
  seed600 + genmarg_zres landed, vdeconv and later items may still be in flight. Run the
  preflight before any submission.
- **Current posteriors (HEAD, fused):** 2D peaks at 0.665 (σ ≈ 0.017; truth 0.73 is
  ~3.7σ away) on both venues; 1D rails onto the 0.60 grid boundary. Combined curves:
  scratchpad `posteriors_head.json` of the 2026-08-29 session (re-derivable from
  `realistic_20260729/headreadout_20260827/*/posteriors*/h_*.json` by Σ log L).

## 1. Entry points

- Charter mirror: §2 here. Leads inventory + posteriors + the six answers: Arc Follow-ups
  artifact (`693acee5`); the narrative: The Hierarchical Arc (`2d90d9d6`); rulings docket
  (`6bfcbba2`).
- Records: `realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md` rows #216–#221;
  `PREREGISTRATION_HIER_HTHETA_20260826.md` (PA-HIER-27..30; §1.2 for the `s` form);
  `PHYSICS_CHANGE_THETA_HOOK_20260828.md`; `PREREGISTRATION_CMEM_READS_20260828.md`;
  `PREREGISTRATION_MKER_WGEOM_20260828.md` (+ RE-ANCHORED EVALUATION);
  `CLAIM_COMPLETION_MEMBERSHIP_20260828.md`; `docs/RESEARCH_CYCLE.md` (amendment ledger).

## 2. The tree (mirror of the charter; depth-1 nodes are covered by row #221)

| branch | 1 (launch under grant) | 2 (fresh [RULE]) | 3 (fresh [RULE]) |
|---|---|---|---|
| **B1 [HIER]** | S0-A mirror null: driver over 4 b0i seeds × 5-node θ-cross, h = 0.73; bands \|Z\| ≤ 3 | S0-B production θ-score at truth, by z + class — **shared instrument with B3** (~75–101 CPU-h) | a: Stage P + C3 build · b: park, redirect budget |
| **B2 [CMEM]** | A1: bc+bt arms, paired within-seed ln-ratio, 10 000 perms, p < 0.01 (free) | A2: cone-widening (k_sky 3) H₀ counterfactual, one venue (~105–265 CPU-h), only if A1 DISPLACED | proposal (out-of-ball term / widened cone) → shared HEAD readout |
| **B3 [POP]** | zero-compute: does row #138's 87 % prediction survive on fused HEAD diagnostics, per z-bin (paired)? | M1-consistent population prior flag (physics gate) + score-at-truth read riding B1.2's arm | adoption proposal → shared HEAD readout; A14 falsifier |
| **B4 [IMP]** | stage-0 intake (both exoneration layers) + information forecast: decompose the ~81 % remainder on banked B-SEL/C-SG tables by impostor z, σ_z, mass, count, in-ball flag | decisive read named by 4.1 (expected mirror counterfactual ≤ 20 CPU-h) | proposal or bound; merge into B1 if the mechanism is the kernel width |
| **B5 [WIN]** | instrument flag `mass_filter_geometry="log"`, k (physics gate, default byte-identical) + zero-compute candidate-count factor on banked cones | registered counterfactual at k = 3: candidate growth + ΔMAP vs HB +0.0015, one venue (~50–130 CPU-h) | adoption gate → shared HEAD readout |
| **B6 [ALIGN]** | gate note + reorder `s` before the PV fold + σ_pv ≠ 0 ordering test → [PHYSICS] commit (must land before S0-B) | — | — |
| **B7 [2D-TWIN]** | proposal: adopt `catalogue_numerator_survival_2d="mz_sel"` (centering decided in-proposal; ×2.5 residual disclosed) | counterfactual arm, one venue (~50–130 CPU-h) | adoption → shared HEAD readout |
| **B8 [CAL]** | F5 information floor at the production venue, 1D + 2D (local) — defines "starved" | build the two-channel calibration harness ([A3]) | coverage + absolute-count audit → stop/continue verdict |

**Waves:** 1 = all depth-1 nodes (local, ≤ 20 CPU-h; one Workflow: 7 sonnet branch workers
+ 1 chair; ≤ 3 top-tier) → synthesis docket 1. 2 = one cluster batch (1.2+3.2, 2.2?, 5.1,
7.2, 4.2; ~350–650 CPU-h; archive-before-run; finish before 09-23) → docket 2. 3 = gates →
ONE blind HEAD readout with per-change arms → 8.3.

**Amendments proposed (need ratification, then append to `docs/RESEARCH_CYCLE.md`):**
F1 one root, many cycles (cross-branch dependency lines) · F2 serialized adoption · F3 shared
instruments register all predictions first · F4 compute ledger + deadline gate · F5 depth
gates + synthesis dockets (branch agents never address the author directly).

## 3. Resume recipe (one line)

Present the charter's three asks → on ratification, launch the wave-1 Workflow (fan-out 8,
sonnet workers, chair synthesis) → synthesis docket 1 → wave-2 batch under the (optional)
standing CPU-h cap → docket 2 → wave 3 → the stop/continue verdict against B8's numbers.

## 4. Standing rules carried (do not re-learn)

Verifier output is evidence, not authority · subagents never run the registered measurement
they built · never end a turn to wait on an untracked process · per-poll SSH, Monitor for
watchers · every submission stamps its authorization · exoneration grep is for the
MECHANISM, not the tag · banked ✓VER rows can be internally inconsistent — one arithmetic
check on a row's own counts before escalating.

## 5. Author rulings after the charter (2026-08-29) — READ BEFORE LAUNCH

- **Charter RATIFIED in full; [STANDING] grant (row #222, verbatim there):** continue through
  every consecutive node of every branch on orchestrator judgement; one synthesis docket per
  wave for INFORMATION; an independent **verifier pass at the end** is the author's check
  (register it as its own workflow: sonnet panel + ≤1 top-tier adjudicator; "refuted" and
  "undetermined" are valued outputs). Assumption pending author confirmation: production
  DEFAULT flips still return to the author with their readout.
- **Chair = an inherit-tier subagent with a scoped context package** (claim-card path,
  exoneration lists verbatim, the branch registration, the node's inputs) — never the session
  orchestrator in-line, never the whole orchestration context. The session orchestrator holds
  the tree, dispatch, the compute ledger and the record. ≤ 3 top-tier agents per wave stands.
- **Cluster:** no CPU-h cap; fairshare is already at the floor (skill gotcha 13) — size every
  arm on need and shape it backfill-friendly; state each wave's total in its launch summary.
- **B5 performance note:** the mass window currently removes only ~4.2 % of cone candidates
  (n_lin/n_all = 0.9577), so k = 3 / log cannot add more than that; the performance risk is
  the SKY cone (B2.2 k_sky 1.5→3 ≈ 4× candidates ≈ 4× kernel cost) — B2.2 must argue its
  size against that scaling.
