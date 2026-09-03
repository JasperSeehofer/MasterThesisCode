# r-sealed-mock — REGISTRATION DRAFT: the sealed-truth anti-tuning mock (redteam T-1)

Date: 2026-09-03 (night). Node: r-sealed-mock (Research Graph 1 ADDENDUM, Branch M [SEAL], §1.4) — **DRAFT**.
Author of record for all scientific decisions: Jasper Seehofer. Prereg author: top-tier subagent (xhigh).
Status: **PROPOSED THROUGHOUT — nothing here is frozen, drawn, generated, or launched.** Authorization:
addendum §3 row A-M1 (authoring runnable tonight under row #325 "continue autonomous, decide but flag").
Every band, the prior, the pool option, the stage (m1) launch and the (m2) cost return as fresh RULE
**d-sealed-register** (§1.6). Research-cycle stages 0–2 applied (stage 3 = the measure nodes, not tonight).
Convention: [DOC] read from a committed/banked file · [LOCAL] recomputed here from local data · [INFER] derived.
Every cap marked ORCHESTRATOR-DERIVED carries its derivation. Append-only after commit.

## 0. Existence contract (three-valued; every input the proposal assumes)

| input | state | evidence |
|---|---|---|
| redteam T-1 text | present | `results/redteam_20260726/PHYSICS_METHODOLOGY_REVIEW.md:401-413` |
| GitHub #39 | present (OPEN, 0 comments, milestone "Paper Submission") | `gh issue view 39` 2026-09-03 |
| ledger preamble §4 item 7 ("ordered, never run") | present | `gate_b_20260730/BIAS_HISTORY_LEDGER.md:218` |
| derivation memo lines on the alternative-truth arm | present | `docs/derivations/realistic_host_observation_model.md:532-533` |
| 0.67 pool CRB (`run_20260729_seed64000_h0p67/`) | **local: absent · cluster: unreachable tonight** (no cluster commands) | local dir `closure_seed64000_h0p67/` holds only `posteriors/` (44 files) + `combined_posterior_2d.json`; `cluster/datasets.yaml:133-138` (commit `7b30d1f`, "UNVERIFIED PROVENANCE beyond run_metadata") |
| 0.67 pool unsealed readouts | present | `IDEALIZED_BASELINE_READOUT.md:25-40` (07-31 zoom: MAP 0.66990, pull −0.24σ, N 1343); `closure_seed64000_h0p67/combined_posterior_2d.json` (mtime 2026-08-03: `map_h 0.67`, `n_events_used 1343`, `variant posteriors_with_bh_mass`, 41 nodes); `posteriors/comparison_table.md` (1D `posteriors` MAP 0.6700 all 4 strategies) |
| 0.77 pool (`run_20260729_seed65000_h0p77`) | local: absent · cluster: unreachable; "no run_metadata" | `cluster/datasets.yaml:135` |
| production p_det pool | present locally (mirror) | `seed61000/simulations/injections/` (707 `injection_h_0p73_task_*.csv`); canonical `cluster/datasets.yaml:32-41` (`h_ref: 0.73`, 500 files, 50 000 events) |
| production CRB seed61000 | present | `seed61000/prepared_cramer_rao_bounds.csv` (1590 rows; md5 of record `9a1f2a14…` per DATA_INVENTORY line 284) |
| production timing anchor | present | `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/` (41 logs + 41 `.out`; `cluster/graph1_headrebaseline_iiib.sbatch:44` `--cpus-per-task=16`) |
| seed61000 simulate task count N_tasks | **absent locally** (0 `run_metadata*.json` under `seed61000/simulations/`); cluster unreachable | NOT-EVALUABLE tonight (§7) |
| sealed-h machinery | absent by construction (grep sealed/h_inj/blind_mock: only the rescaling scripts) | addendum §1.4; `scripts/bias_investigation/test_17_rescale_crb_to_h065.py`, `cluster/evaluate_closure_h065.sbatch` |

## 1. Question, claim, provenance (stage 0)

**q-anti-tuning** (addendum §1.0): does the current production stack recover a SEALED h_inj ≠ 0.73?
Claim node c-no-tuning-to-0.73: "supported by inspection only" (#39 body: "no numerical anchor to 0.73")
+ one unsealed pre-flip closure. T-1 verbatim [DOC]: "Regenerate a full venue at h_inj = 0.68 (or, better, at
a value drawn blind by a script and sealed until readout) with the same catalogue, same completeness cache,
same population model, and a freshly generated injection pool at that cosmology. Run the *unchanged*
production stack on a 41-point grid spanning the new truth. Pre-register: MAP within 2×(empirical seed
scatter) of 0.68, interior, plus the T-4 pull test." The cheap d_L-rescaling variant is "NOT a substitute"
(`:409-413`). **Refute by:** a recovered posterior concentrated at the 0.73-production value (0.666) whatever
h_inj is — the G7d cell of `cluster/evaluate_closure_h065.sbatch:12` ("MAP ≈ 0.730 → pipeline TUNED to 0.73;
HALT paper"), re-anchored below.

## 2. Why the existing 0.67 closure is neither sealed nor current [DOC]

1. **Unsealed:** the truth is in the directory name (`run_20260729_seed64000_h0p67`), in every
   `run_metadata_*.json` (`cli_args.h_value`; cf. the retrieved evaluate run's `run_metadata_0.json`), and in
   the closure readouts quoted in §0. Nobody who ran, read, or fixed the estimator between 07-29 and today was
   blind to it.
2. **Single p_det pool at h_ref = 0.73:** `cluster/datasets.yaml:32-38` ("a single-h pool suffices"); the 0.67
   simulate used it (submit_pipeline.sh threads only `H_VALUE` to simulate, `:222-227`).
3. **Pre-flip stack:** simulated at `7b30d1f` (2026-07-29), evaluated 07-31/08-03. Six production
   `[PHYSICS]` flips since (ORCHESTRATOR-DERIVED list from `docs/gates/PHYSICS-GATE-LEDGER.md` + ledger; row
   #328 counts "six"): Σ^φ divisor `e35ea018` (row #179); catalogue-leg twin `bac48696` (row #195/#197);
   [P3-WBHZERO] `cf4f8a2a` (row #202); 2D twin default mz_sel/eff (A14, ratified row #284); 1D mass-aware
   `catalogue_leg_1d_mass_aware="auto"` (row #286); h-grid decoupling `a26959b4` (row #313). The 0.67
   readouts test none of them.
4. **Lossy pool:** GPU tasks hit the 30-min wall, "~10 of 40 requested steps per task"
   (`REALISTIC_READOUT.md:180-185`); 1343 events vs ~1590 (`IDEALIZED_BASELINE_READOUT.md:36-40`).

## 3. The p_det pool at h_inj — a registered disclosure, not a decision [DOC]/[INFER]

T-1 demands a fresh pool at h_inj. The estimator's own docstring claims the pool is **h-invariant by
construction**: "each injection has an h-invariant detection horizon d_hor_k = SNR_k·d_L_k/snr_threshold …
The horizon set is independent of the trial Hubble parameter h … the survival grid is built once and reused
for every h" (`darksiren_emri/bayesian_inference/simulation_detection_probability.py:12-24`; same claim
`cluster/datasets.yaml:32`, `test_17…:53-58`). If exact, a fresh pool adds pooled samples and nothing else;
if not exact, the fresh pool is the only test of it. Two registered options, both returned to the author:

| option | content | cost | what it buys |
|---|---|---|---|
| **P1 (T-1 verbatim)** | fresh 500-task pool at h_inj, sealed | ≤ 250 GPU-h ORCHESTRATOR-DERIVED: 500 tasks (`datasets.yaml:39`) × 0.5 h wall cap (`cluster/inject.sbatch:19`) | the selection model exercised at h_inj as T-1 asks; **leak hazard**: file names carry h (`injection_h_(\d+p\d+)` parsed at `simulation_detection_probability.py:316`) and the `h_inj` column is loaded (`:165,:350`) — a naming shim (new code, no physics) is required to keep the pool sealed |
| **P2 (invariance gate + reuse)** | a SMALL unsealed pool at a public h (e.g. 0.67, 20 tasks ≈ 10 GPU-h) compared to the rescaled 0.73 pool's survival grid (max abs Δp_det on the grid ≤ 1e-3 registered); the sealed run then reuses the canonical 0.73 pool | ≈ 10 GPU-h ORCHESTRATOR-DERIVED (20 × 0.5 h) | tests the h-invariance claim directly, at 1/25 of P1; leaves T-1's literal "at that cosmology" unmet |

RECOMMENDATION (flagged, not a ruling): **P2 first**; P1 only if the invariance gate fails.

## 4. Sealed commitment mechanism (DESIGN; nothing executed)

What the seal can and cannot do, stated plainly: **the anti-tuning guarantee is the CODE FREEZE, not the
secrecy of the number.** h_inj is recoverable by anyone who opens the run directory (leak inventory below);
the commitment proves *when* the value was fixed relative to the frozen code, and the procedural blindness
keeps the reading agents honest. Order of operations (each step = one ledger row or one file in this dir):

1. **Code freeze FIRST.** The production commit hash H_code is written into this registration (§10 slot)
   BEFORE any draw. The sealed sbatch uses **strict equality** `git rev-parse HEAD == H_code` (the
   ancestor-check pattern of `cluster/graph1_m_s0b_production.sbatch:83-86` is NOT acceptable here — it
   would admit later flips into a sealed run).
2. **Draw** (`scripts/sealed_mock/draw_sealed_h.py`, stdlib only, DESIGN): reads the registered prior from
   this file's §5 slot; `h_inj = lo + (hi−lo)·u`, u from `secrets.SystemRandom()` (NOT a seeded RNG — a public
   seed is a leak); `salt = secrets.token_hex(32)`; `commitment = sha256(f"{h_inj:.6f}|{salt}")`. Writes
   `SEALED_H_INJ.json {h_inj, salt, prior, H_code, utc}` to `~/.sealed/r-sealed-mock/` (outside the repo,
   outside every agent cwd, mode 0400), prints ONLY the commitment. The commitment + prior + H_code go into
   the ledger row that launches (m2).
3. **Submit** (`cluster/submit_sealed_pipeline.sh`, DESIGN): reads the sealed file at submit time, exports
   `H_VALUE` to `simulate.sbatch` (`:81`), job names free of h, `RUN_DIR` named by the commitment prefix.
   Unavoidable leaks INSIDE the run dir: `simulate.sbatch:91` echoes `H_VALUE (truth)`; `main.py:406` names the
   log `…_h_<h_value>.log`; every `run_metadata_*.json` stores `cli_args.h_value`; the merged CRB stores
   `luminosity_distance` next to `host_galaxy_index` (76-class in-catalogue rows ⇒ h from the catalogue z).
   **Procedural rule:** until the unseal row, no agent opens `RUN_DIR/logs`, `run_metadata_*`, the CRB, or the
   sealed file; the read node consumes `simulations/posteriors*/h_*.json` ONLY, through a script that never
   prints a CLI arg; its output is the pull table of §6 with h_inj substituted by the sealed file at unseal.
4. **Unseal** = one ledger row quoting the commitment, the file's `{h_inj, salt}`, the recomputed hash, and
   the pre-computed posterior moments; the cell assignment (§6) is mechanical from that row.
5. **max_revisions:** the SEALED measurement (m2) has revision cap 1 — a seal cannot be re-drawn without
   destroying it (addendum §1.7). The chair's launch note asked for max_revisions 2; registered here as: design
   revisions of this un-drawn registration ≤ 2 (no draw has happened, nothing is destroyed), sealed draw ≤ 1.
   **Discrepancy routed to d-sealed-register.**

## 5. Grid, prior on h_inj, and why the proposal's U[0.62, 0.84] is not launchable as written [DOC]/[LOCAL]/[INFER]

- Grid of record: `H_GRID_41` = [0.600, 0.860] (`cluster/graph1_headrebaseline_iiib.sbatch:89`); the G-EXT
  wing to 1.00 is admissible (`a26959b4`; row #313) but the host-window bound `h.upper_limit` stays 0.86 and
  the entry guard rejects h < `h.lower_limit` = 0.6 (`cosmological_model.py:400-401`;
  `bayesian_statistics.py:4677` "Hubble constant out of bounds" guard). **No node below 0.60 is evaluable without a
  downward [PHYSICS] decoupling (mirror of a26959b4).**
- Production anchors (row #302, re-baseline iiib): 2D mean 0.665854, σ 0.018475; 1D mean 0.666987, σ 0.017526;
  offsets −0.0641 / −0.0630 (mean − 0.73).
- Separation requirement (ORCHESTRATOR-DERIVED): the TUNED cell (mean ≈ 0.666 whatever h_inj) and the
  TRANSFER cell (mean ≈ h_inj − 0.064) differ by |h_inj − 0.73|; at ≥ 3σ_h with σ_h ≈ 0.020 at reduced N
  ⇒ **|h_inj − 0.73| ≥ 0.06–0.07**. The proposal's U[0.62, 0.84] puts 0.14/0.22 = **64 % of its mass in the
  dead zone** [0.66, 0.80] (proposal §1.4, "ORCHESTRATOR-DERIVED", never checked against σ_h).
- Low side: TRANSFER at h_inj ≤ 0.66 ⇒ mean ≤ 0.596 < 0.60 floor ⇒ railed/censored (g-censoring red).
  **The whole low half is un-evaluable on the current grid.**
- High side: the host z-window is built from the h-bounds [0.6, 0.86]; at h_inj the true host sits
  (0.86 − h_inj)/0.86 in d_L inside the window's upper-h edge. Production σ_frac = σ_dL/d_L: median 0.0373,
  p16–p84 0.021–0.049 (`seed61000/prepared_cramer_rao_bounds.csv`, [LOCAL]). Margin in σ_frac(median):
  0.84 → 0.6σ (truncated for typical events — not production-like), 0.82 → 1.25σ, 0.80 → 1.9σ, 0.79 → 2.2σ.
- Detection loss at higher h (SNR ∝ 1/d_L ∝ h at fixed z): fraction of the 1590 production events with
  SNR·0.73/h_inj < 20 — 15.5 % at 0.80, 22.6 % at 0.84 [LOCAL] ⇒ N ≈ 1230–1340, σ_h × 1.09–1.14.

**Registered prior (slot; RECOMMENDATION, flagged): P-A = Uniform[0.79, 0.82].** Separation from TUNED
0.06–0.09 = 3.0–4.5σ at σ 0.020; TRANSFER mean 0.726–0.756 and UNBIASED mean 0.79–0.82 both interior on
the 55-node grid with ≥ 2σ to 0.86 and the wing beyond; window margin 1.25–2.2σ_frac (disclosed weakness).
**Alternative P-B** = two-sided U([0.62, 0.66] ∪ [0.79, 0.82]) requires the downward decoupling first (a
[PHYSICS] build node + author word) — returns with its own cost. The one-sidedness of P-A is public and
harmless: the code is frozen before the draw.

## 6. Statistics, hypothesis cells, pull band (ORCHESTRATOR-DERIVED; anchors row #302)

Per channel c ∈ {2D primary (`posteriors_with_bh_mass`), 1D replicate}, venue iiib (production catalogue),
frozen T0 gradient-weighted scorer (row #284 [DO]; the m-head-rebaseline convention): mean_h,c, MAP_c, σ_h,c
on the 55-node grid; **pull_c = (mean_h,c − h_inj)/σ_h,c**. joint_r1 = optional replicate (word to the author).

| cell | criterion (2D primary; 1D must agree in cell) | reading |
|---|---|---|
| **TUNED** | \|mean_h − 0.6659\| ≤ 3σ_h AND \|MAP − 0.665\| ≤ 3σ_h, for h_inj with \|h_inj − 0.73\| ≥ 0.06 | the G7d cell: posterior anchored to the 0.73-production value ⇒ HALT paper claims; fresh RULE |
| **TRANSFER** | \|(mean_h − h_inj) − (−0.0641)\| ≤ 3σ_h AND MAP inside the same window | the −0.064 offset is h-independent: the anti-tuning mechanism check PASSES; the additive/multiplicative split (Δ ≤ 0.006 across P-A) is REPORTED-ONLY, not separable |
| **UNBIASED-AT-h_inj** | \|mean_h − h_inj\| ≤ 2σ_h | the 0.73 offset is 0.73-specific — itself anomalous; fresh RULE (separated from TRANSFER by 3.2σ only, disclosed) |
| **INTERMEDIATE** | none or several of the above; 1D/2D disagree | banked, fresh RULE; revision cap per §4.5 |
| **NO-READ** | g-censoring red (MAP at a grid edge; edge-node mass > 1e-3), g-population red, pins/freeze red | nothing banked |

Secondary (reported, not cell-bearing): the T-4 per-event pull calibration (`PHYSICS_METHODOLOGY_REVIEW.md:433-441`)
on events with curvature > 0; the seed-scatter caveat: only two 0.73 truth seeds exist (paired difference
−0.023, `REALISTIC_READOUT.md:51-62`) — a single sealed seed cannot separate seed scatter from transfer error
beyond the 3σ_h bands above.

## 7. Stages, inputs, costs (every number sourced; caps ORCHESTRATOR-DERIVED)

**Timing anchor [LOCAL]:** the retrieved 41-task iiib HEAD re-baseline (N = 1588): per-task wall from
log-name start stamp to `.out` mtime = 5.57–6.60 min, median 6.27, Σ = 4.196 wall-h; 16 cores/task
(`graph1_headrebaseline_iiib.sbatch:44`) ⇒ **67.1 core-h**. Consistency: row #285's "≈94 CPU-h" = 55 tasks ×
16 × 6.4 min ✓; the proposal's "9" = 2 venues × ~4.5 WALL-h (candidate 11 quotes wall-hours as CPU-h —
a unit conflation, disclosed). All caps below are core-hours (the ledger's convention).

| stage | inputs | cost | cap |
|---|---|---|---|
| **(m1) HEAD re-score of the 0.67 pool** (both channels, iiib, `H_GRID_41`) | cluster CRB of `run_20260729_seed64000_h0p67` (md5 pinned at retrieval; STOP on mismatch), the canonical 0.73 pool, catalogue md5 `c52c13b5…`; `/cluster` preflight READY | 41 tasks × 16 × (5.57–6.60 min) = 61–72 core-h at N = 1588; × 1343/1588 if linear in N ⇒ **52–72 core-h** (≈ 3.5–4.5 wall-h of array) | 75 core-h iiib; +75 if joint_r1 |
| **(m2) sealed pool** | P1 or P2 (§3); simulate at h_inj; evaluate 55 nodes both venues | pool ≤ 250 GPU-h (P1) or ≈ 10 GPU-h (P2); simulate **N_tasks × ≤ 0.5 GPU-h** (`simulate.sbatch:36-38`; N_tasks NOT-EVALUABLE locally — read `run_20260729_seed61000` task count / `sacct` for 6090909–6090912 before the ruling); evaluate 55 × 16 × 6.3 min ≈ 92 core-h/venue at N ≈ 1588, ≈ 72–80 at the P-A N | **94 core-h/venue** (row #285 anchor) + the GPU items |

**(m1) evaluability disclosure [INFER]:** under TRANSFER the 0.67 pool's expected mean is 0.606 (additive) /
0.611 (multiplicative) — within 0.5σ_h of the 0.60 floor ⇒ the TRANSFER/UNBIASED split is NOT-EVALUABLE on
`H_GRID_41`; the TUNED cell IS evaluable (0.666 sits 3.3σ above the floor). (m1) therefore registers only the
binary **TUNED vs NOT-TUNED** (posterior mass within 3σ_h of 0.666 vs concentrated at/below 0.63), with the
rail disclosed. The 0.77 pool (if its provenance can be recovered) would give TRANSFER 0.702–0.706 vs TUNED
0.666 — 1.9σ, REPORTED-ONLY. Both facts argue for (m2) with P-A rather than more (m1)-class reads.

## 8. Gates (scored before any cell is read; red ⇒ NO-READ)

- **G-1 pins:** catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; the 0.73 pool path + file count 500; the
  0.67 CRB md5 recorded at retrieval; the sealed CRB md5 recorded at merge (never printed with h).
- **G-2 code freeze:** `GIT_COMMIT_AT_RUN.txt` == H_code, strict; `provenance_*.json` dirty count 0 for the
  sealed run (the re-baseline ran dirty=476 — acceptable for it, not here).
- **G-3 commitment:** at unseal, sha256 recomputed from the file equals the ledger-quoted commitment.
- **G-4 blindness audit:** grep of every agent record written between draw and unseal for the sealed value
  (to 3 decimals) returns nothing; the read script's whitelist (`posteriors*/h_*.json`) is the only path.
- **g-population:** simulate log count vs CRB rows; the timeout yield per task disclosed (the 0.67 precedent);
  N at h_inj vs the 15–23 % SNR-loss expectation of §5.
- **g-censoring:** MAP interior; posterior mass on the two outermost nodes < 1e-3 (row #313 wing convention).
- **g-znorm / g-precision:** as the parent panel (row #290); full-precision columns; no 7-s.f. column.
- **P2 invariance gate (if P2):** max |Δp_det| on the (d_L, M) grid between the small h-pool and the rescaled
  0.73 pool ≤ 1e-3; failure ⇒ P1 becomes mandatory.

Invariants ([A10]): H_code · catalogue · population model (Barausse M1 cosmology, CLAUDE.md bug 8) · the
completeness cache · PRODUCTION_FLAGS verbatim · the T0 scorer · both md5 pins. **Structural blindness:** the
test cannot detect tuning that is h-INVARIANT (a normalisation defect biasing every truth by the same
−0.064 lands in TRANSFER and is *classified*, not exonerated); it shares the population model with
production (a common-mode defect there is invisible); one seed ⇒ seed scatter is unmeasured at h_inj.

## 9. Disposition table (every row returns as a fresh RULE; verdict-free here)

| stage | outcome | consequence proposed to the author |
|---|---|---|
| (m1) | NOT-TUNED (mass concentrated below 0.63 or at a rail) | the 0.67 pool refutes the G7d cell on the post-flip stack; the anti-tuning stamp stays PARTIAL (unsealed, TRANSFER unresolved) |
| (m1) | TUNED | HALT paper claims; (m2) still runs (the sealed test is the verdict-bearing one) |
| (m2) | TRANSFER | T-1 PASSED; c-no-tuning-to-0.73 → VERIFIED; the anti-tuning stamp attaches to d-paper-1d2d-verdict / d-paper-coverage (STANDING A-S) |
| (m2) | TUNED | undetected tuning dependency; every iiib/joint_r1 verdict re-scoped; HALT |
| (m2) | UNBIASED-AT-h_inj | the −0.064 offset is 0.73-specific ⇒ new stage-0 claim; fresh RULE |
| (m2) | INTERMEDIATE / NO-READ | banked; the seal is spent (cap 1) — a second sealed draw is a new registration |

## 10. Freeze slots (filled only by the launching ledger row, never by this draft)

H_code = ______ · prior = ______ · pool option = ______ · commitment = ______ · draw UTC = ______.

## 11. Open questions routed to d-sealed-register (numbered; RECOMMENDATIONs flagged, none binding)

1. Prior: P-A U[0.79, 0.82] (RECOMMENDED) vs P-B two-sided (needs the downward decoupling build node).
2. Pool: P2 invariance gate + reuse (RECOMMENDED) vs P1 fresh sealed pool (≤ 250 GPU-h + naming shim).
3. max_revisions: proposal §1.7 says 1, the launch note said 2 — registered as design ≤ 2 / sealed draw ≤ 1.
4. (m1) binary-only read on `H_GRID_41` (RECOMMENDED: accept) vs funding the downward decoupling to resolve
   TRANSFER on the 0.67 pool; the 0.77 pool as a REPORTED-ONLY replicate (its provenance is weaker).
5. N_tasks for the sealed simulate (a cluster read) and scheduling vs the 2026-09-23 workspace expiry
   (docket 12b); the (m2) evaluate on 55 nodes (94 core-h/venue cap) vs 41 (67).
6. joint_r1 replicate for (m1)/(m2): yes/no (doubles the evaluate cost).
7. Whether the read node is a top-tier decisive verifier (proposal §5.3: yes for the unseal read).

Cost tonight: 0. Tiering (addendum §5.3): (m1)/(m2) arrays sonnet/low; unseal read top-tier.
