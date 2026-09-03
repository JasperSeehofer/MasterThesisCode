# d-sealed-register — decision dossier (batch 2, 2026-09-04) + the stage (m1) launch block

Node: d-sealed-register (Research Graph 1 ADDENDUM §1.6, Branch M [SEAL]). Author of record:
Jasper Seehofer. Assembled by prereg author B (top-tier, xhigh) from
`exec/r-sealed-mock/REGISTRATION_DRAFT.md` (2026-09-03 night) and
`GRAPH1_ADDENDUM_PROPOSAL_20260903.md` §1.4/§1.6/§1.7/§3 (A-M1..A-M3)/§5.3.

**Status of every reading below: CHAIR-PROVISIONAL under the row #349 batch-2 grant
("ratify-and-flag"), veto-able by the author at the next message. Nothing here is an author
ruling.** Authorization chain, quoted: row #349 (2) *"the docket-item-2.2 STANDING (launch inside
caps after a green gate) is read as RENEWED for this batch by the author's cluster words"*;
row #349 (3) item (2) *"sealed-mock stage m1 — HEAD re-score of the existing
closure_seed64000_h0p67 pool, both venues if cost allows (52–72 core-h/venue, cap 75/venue),
behind a d-sealed-register dossier ratified-by-chair (flagged)"*; docket R10 provisional
reading *"A-M1 prior P-A = U[0.79,0.82] as registered"*; addendum §3 rows A-M1 (Approved:
authoring), A-M2 (Approved: the (m1) re-score "after d-sealed-register, behind /cluster
preflight READY"), A-M3 (Approved: (m2) "after d-sealed-register; scheduling vs the 2026-09-23
expiry is the author's word").

**Design-gate record: ABSENT.** Addendum §1.6 lists the requires-manifest of d-sealed-register
as "r-sealed-mock design-gate record"; §1.8 planned a 2-member sonnet design-gate panel; no such
record exists under `exec/r-sealed-mock/` (only `REGISTRATION_DRAFT.md`, 2026-09-04 listing).
The docket-2.2 STANDING requires "a green gate" before launch. **This is the first blocker of
§4** — the (m1) sbatch is written and ready but is NOT launchable under the STANDING until a
design gate is run and GREEN (a sonnet panel, zero compute; the chair may run it tonight).

---

## 1. What the register node asks (every item = one line the author can answer)

| # | decision | draft's RECOMMENDATION | chair provisional reading (batch 2) | covered by batch 2? |
|---|---|---|---|---|
| 1 | **Prior on h_inj** | P-A = U[0.79, 0.82] (separation from TUNED 3.0–4.5σ_h at σ ≈ 0.020; both TRANSFER and UNBIASED means interior with ≥ 2σ to the 0.86 bound; window margin 1.25–2.2 σ_frac disclosed as the weakness). P-B two-sided needs a downward [PHYSICS] decoupling first | **ratify-and-flag P-A** (docket R10 reading) | registration text only — the DRAW is NOT covered (§3) |
| 2 | **p_det pool option for (m2)** | P2 (small unsealed pool at a public h + invariance gate `max|Δp_det| ≤ 1e-3` + reuse of the canonical 0.73 pool, ≈ 10 GPU-h) first; P1 (fresh sealed 500-task pool ≤ 250 GPU-h + naming shim) only if P2 fails | **not read** — (m2) is outside batch 2 | NO |
| 3 | **max_revisions** | design revisions of the un-drawn registration ≤ 2; sealed draw ≤ 1 (proposal §1.7 says 1; the launch note said 2 — discrepancy routed here) | **ratify-and-flag: design ≤ 2 / draw ≤ 1** (the draft's own reconciliation; consistent with §1.7's "a sealed test cannot be re-drawn") | yes (registration text) |
| 4 | **(m1) read on `H_GRID_41`: binary only** | accept: TUNED vs NOT-TUNED only; the TRANSFER/UNBIASED split is NOT-EVALUABLE on this grid (expected TRANSFER mean 0.606/0.611 within 0.5σ_h of the 0.60 floor; g-censoring red by construction on that branch) | **ratify-and-flag: binary read, rail disclosed** (§2 below) | yes — this IS the batch-2 launch |
| 5 | **N_tasks for the sealed simulate; scheduling vs 2026-09-23 expiry; 55 vs 41 nodes for (m2)** | read N_tasks from `run_20260729_seed61000` on the cluster before ruling | **not read** | NO |
| 6 | **joint_r1 replicate** | word to the author | **(m1): iiib FIRST; joint_r1 ONLY if the iiib array's measured cost lands ≤ 60 core-h** (row #349 "both venues if cost allows"; 60 = 75-cap minus the joint_r1 ≥ 2.2× wall factor's risk margin — ORCHESTRATOR-DERIVED: at ≤ 60 core-h iiib the joint_r1 sibling (same array, ≥ 2.2× per task per `wave3_headreadout_joint_r1.sbatch`) still fits its own 75 cap only if the iiib per-task wall ≤ 6.2 min; the ops agent reads `sacct` before deciding) `[FLAG]`; (m2): not read | (m1) yes, conditional |
| 7 | **Unseal read tier** | top-tier decisive verifier (proposal §5.3) | **ratify-and-flag** | registration text only |
| 8 | **Leak inventory + procedural rule** (draft §4 item 3) | accept as the registered blindness protocol: no agent opens `RUN_DIR/logs`, `run_metadata_*`, the CRB or the sealed file between draw and unseal; the read script whitelists `posteriors*/h_*.json` | **ratify-and-flag** — and note it does NOT apply to (m1), whose truth (0.67) is public (draft §2) | yes |
| 9 | **Cost caps** | (m1) 75 core-h/venue; (m2) 94 core-h/venue evaluate + GPU items | **(m1) cap 75 core-h/venue ratified-and-flagged (row #349 wording); (m2) not read** | (m1) only |

### 1.1 The (m1) leak inventory (quoted from the draft, so the (m1) reader knows what it may open)
The 0.67 pool is UNSEALED by construction: truth in the directory name
(`run_20260729_seed64000_h0p67`), in every `run_metadata_*.json` (`cli_args.h_value`), in the
07-31/08-03 readouts (`IDEALIZED_BASELINE_READOUT.md:25-40`: N 1343, MAP 0.66990 on the zoom;
local `closure_seed64000_h0p67/posteriors/comparison_table.md`: MAP 0.6700 all 4 strategies,
1343 events; `combined_posterior_2d.json` map_h 0.67). **Consequence (registered):** (m1) is a
mechanism check on a public truth — explicitly NOT the T-1 verdict (addendum §1.0: "partial,
unsealed; explicitly NOT the T-1 verdict"). The blindness that (m1) DOES have is that the
post-flip HEAD stack has never been run on this pool (six production flips since 7b30d1f, draft
§2 item 3) and its result is unknown to everyone.

---

## 2. The stage (m1) registration as launched (zero fresh choices; every number sourced)

**Object.** `m-closure067-headstack` (addendum §1.4): the existing 0.67 pool re-scored on the
current production default (post-flip, post-A14), 41 nodes, both channels, venue iiib.

**Data the job reads (cluster paths; the ops agent verifies existence before submission):**
- CRB of the 0.67 pool: `$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv`
  (+ `cramer_rao_bounds.csv`); `$WS = /pfs/work9/workspace/scratch/st_ac147838-emri`
  (`cluster/datasets.yaml:131-138`, id `campaign51_seed6x000`, commit `7b30d1f`, "UNVERIFIED
  PROVENANCE beyond run_metadata"; jobs 6090909–6090912 per `REALISTIC_READOUT.md:180`).
  **Exists ONLY on the cluster** — the local mirror `results/campaign51_20260728/realistic_20260729/
  closure_seed64000_h0p67/` holds `posteriors/` (44 files) + `combined_posterior_2d.json` only,
  NO CRB (grep of `results/`, `cluster/`, `docs/` 2026-09-04). No rsync-UP is needed or possible
  (nothing local to push); the pin is taken at first touch on the cluster (below).
- The p_det pool: the canonical `$WS/injection_pool_depth15_50k` (`datasets.yaml:33-41`, h_ref
  0.73, 500 files, "CURRENT (campaign canonical)") — the pool the 0.67 simulate itself used
  (draft §2 item 2: "single p_det pool at h_ref = 0.73"). STOP-gated on file count 500 (draft
  §8 G-1: "the 0.73 pool path + file count 500"). The job ALSO records where
  `run_20260729_seed64000_h0p67/simulations/injections` resolves, and STOPs if that link exists
  and resolves elsewhere (a pool mismatch would be a fresh choice nobody made).
- Reduced catalogue `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`, md5
  `c52c13b5cab61f6b3f04bbe202550969` (STOP-gated, as in every head-rebaseline task).

**The md5 pin for the 0.67 CRB: UNKNOWN LOCALLY — pin-at-first-touch.** No md5 of
`run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv` exists in
`DATA_INVENTORY.md`, `datasets.yaml`, or any record (grepped). Registered procedure (draft §8
G-1 "the 0.67 CRB md5 recorded at retrieval"): the ops agent runs, BEFORE submission,
```
ssh bwunicluster 'WS=$(ws_find emri); f=$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv; ls -l $f; md5sum $f; wc -l $f; ls $WS/run_20260729_seed64000_h0p67/simulations/ ; readlink -f $WS/run_20260729_seed64000_h0p67/simulations/injections 2>/dev/null'
```
writes the md5 into this dossier's §5 pin slot AND passes it as `EXPECTED_CRB67_MD5=<md5>` on
the sbatch `--export` line; the sbatch STOPs if the variable is unset or mismatched. The row
count is recorded next to it (expected 1343 data rows = the events used by every 07-31/08-03
readout; a different count is a g-population disclosure, not a STOP — the draft registers
g-population as a read-time gate). If `prepared_cramer_rao_bounds.csv` is ABSENT (only
`cramer_rao_bounds.csv` present), STOP and report — `scripts/prepare_detections.py` would have
to be run first, which is a fresh step nobody registered.

**CLI (the production CoR-P set, verbatim from `cluster/graph1_headrebaseline_iiib.sbatch`,
= `REGISTERED_RESOLVED_FLAGS` in `darksiren_emri/validation/correspondence_1d.py`):**
`--evaluate --h_value <h> --seed 777000+TID --simulation_index TID --strategy physics-floor
--pdet_dl_bins 60 --pdet_mass_bins 40 --pdet_estimator local_linear --pdet_z_resolved
--fisher_cond_threshold 1e16 --host_z_kernel volume_deconv --host_mass_kernel auto
--normalization_mode absolute_marginal --selection_in_completion_numerator fused
--catalogue_mass_overlap production --catalogue_mass_error_scale 1.0 --completion_b_scale derived
--eddington_m on --sigma4d_mass_kernel point --completion_event_measure ratio
--catalogue_global_selection phi --mass_filter_geometry linear --mass_filter_k 1.5 --theta_b 0.0
--theta_s 1.0 --theta_sites all --log_level INFO` — BLIND to the post-flip flags exactly as the
rebaseline (no `--catalogue_numerator_survival_2d*`, `catalogue_leg_1d_mass_aware` left at
`auto` → resolves `on` post-flip; `run_metadata_*.json` records the resolved values; the
resolved-flag equality gate of row #347 (13 tokens vs `REGISTERED_RESOLVED_FLAGS`) is applied at
read time). Grid: `H_GRID_41` (0.600–0.860), one h per array task, `SLURM_ARRAY_TASK_ID` = the
canonical H41 index (task 21 = 0.730). Seeds `EVAL_SEED 777000 + TID` (the rebaseline's own
convention, gotcha 4). Resources copied: `cpu_il`, 16 cpus, `--time=00:45:00`, `--array=0-40`.
HEAD pin: ancestor check on `081b1f28` + `darksiren_emri/` diff-quiet vs `081b1f28` (the
physics is frozen at the S0-B/batch-2 state; `GIT_COMMIT_AT_RUN.txt` records H_code — for (m1)
this is the "code freeze" of draft §4 item 1 in its ancestor form; the STRICT-equality form is
reserved for the sealed (m2) run, as the draft requires).

**Out-root:** `$WS/graph1_sealed_m1_iiib_20260904` (fresh; absent-verified before submission).
**Venue switch:** `VENUE=iiib` (default). `VENUE=joint_r1` adds `--observed_catalogue
$WS/realizations_20260729/observed_catalogue_seed900001.csv` with its sha256 pin
`e8f7ab31…f6751` and the sidecar `parent_csv` check (gotcha 10), out-root
`$WS/graph1_sealed_m1_joint_r1_20260904`, `--time=01:30:00` — verbatim the
`graph1_headrebaseline_joint_r1.sbatch` delta. Submitted ONLY under §1 item 6's condition.

**Statistic and reading rule (draft §6/§7, T0 gradient-weighted scorer of record — row #284
[DO]; `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` convention as used by
`exec/m-head-rebaseline/READOUT_RECORD.md`):** per channel c ∈ {2D `combined_with_bh` primary,
1D `combined_no_bh` replicate} on `simulations/diagnostics/event_likelihoods.csv` (41 h × N
events): `logpost(h) = Σ_e log L_e(h)` (physics-floor zero handling, uniform prior), `w =
np.gradient(H_GRID_41)`, `mean_h = Σ post_n(h)·h·w(h)`, `σ_h` likewise, `MAP = argmax`.
Anchors (row #302, iiib re-baseline at truth 0.73): 2D mean 0.665854 / MAP 0.665 / σ 0.018475;
1D mean 0.666987 / MAP 0.665 / σ 0.017526. Registered binary (draft §7):

| cell | criterion (2D primary; 1D must agree in cell) | reading |
|---|---|---|
| **TUNED** | `|mean_h − 0.6659| ≤ 3σ_h` AND `|MAP − 0.665| ≤ 3σ_h` (1D: `|mean_h − 0.6670| ≤ 3σ_h`, `|MAP − 0.665| ≤ 3σ_h`) | the G7d cell on the post-flip stack: HALT paper claims; fresh RULE; (m2) still runs |
| **NOT-TUNED** | posterior mass concentrated at/below 0.63 (`|mean_h − 0.6659| > 3σ_h` with `mean_h < 0.6659`) — INCLUDING a railed posterior (MAP at 0.600 or edge-node mass > 1e-3), which is booked NOT-TUNED-AT-RAIL and read AS A BOUND per g-censoring's own convention (panel line 245) | the 0.67 pool refutes G7d on the post-flip stack; the anti-tuning stamp stays PARTIAL (unsealed; TRANSFER unresolved on this grid) |
| **INTERMEDIATE** | neither (e.g. `mean_h > 0.6659 + 3σ_h`, or 1D/2D disagree in cell) | banked, fresh RULE |
| **NO-READ** | pins/freeze red; g-population red (N vs 1343 undisclosed); resolved-flag equality red | nothing banked |

The brief's comparison `|mean_h − 0.67|` vs `|mean_h − 0.666|` is REPORTED alongside (both
distances in units of σ_h, per channel): it is the same fork as TUNED vs "near truth", but on
`H_GRID_41` "near 0.67" is NOT the expected NOT-TUNED location (draft §7: TRANSFER predicts
0.606–0.611), so it is reported, not cell-bearing. Bands are the registered 3σ_h (draft §6)
throughout; σ_h is the measured one per channel at the pool's N (1343 ⇒ σ_h × ≈1.09 vs 1588,
draft §5 last bullet).

**Gates at read time (draft §8, (m1) subset):** G-1 pins (CRB md5 = the §5 slot; pool file
count 500; catalogue md5); G-2 `GIT_COMMIT_AT_RUN.txt` contains `081b1f28` (ancestor form);
g-population (CRB rows vs 1343; the pool's lossy-task disclosure "~10 of 40 requested steps per
task", `REALISTIC_READOUT.md:180-185`); g-censoring (MAP interior; mass on the two outermost
nodes < 1e-3; a red here is a BOUND, not NO-READ — see the NOT-TUNED row); g-znorm / g-precision
as the parent panel (full-precision columns; the g-znorm identity is NOT evaluable from this
output type — the head-rebaseline reader found no `global_denom_*` columns; disclosed, not
skipped); resolved-flags equality (row #347's 13-token check vs `REGISTERED_RESOLVED_FLAGS`).

**Cost (draft §7 [LOCAL] anchor):** 41 tasks × 16 cores × 5.57–6.60 min = 61–72 core-h at
N = 1588; × 1343/1588 if linear in N ⇒ **52–72 core-h** (≈ 3.5–4.5 wall-h of array). **Cap 75
core-h iiib** (+75 joint_r1 if §1 item 6 fires). Reservation basis (elapsed × 16) — the same
basis as the draft; note the S0-B lesson (row #336 (6)) that elapsed × cores-used would read
≈ 5× lower; the cap here is on the reservation basis by the draft's own convention.

**Invariants (A10):** H_code (ancestor of `081b1f28`, `darksiren_emri/` diff-quiet), catalogue
md5, the canonical 0.73 pool, the Barausse M1 population model, the completeness cache,
`REGISTERED_RESOLVED_FLAGS` verbatim, the T0 scorer, `H_GRID_41`. **Structural blindness:** an
h-INVARIANT tuning (a normalisation defect biasing every truth by the same −0.064) is
classified TRANSFER-like, not exonerated; the pool is lossy (1343 of ~1590) and its
timeout-selection is uncharacterised at h = 0.67 (Branch L's object); one seed; the truth is
public (no anti-tuning claim can bank from (m1)).

**Read node:** a fresh sonnet reader re-derives from the raw CSV; chair re-derives every
decisive number (evidence, not authority). The reader does NOT run the registered measurement.

---

## 3. NOT covered by the batch-2 grant (each returns to the author as its own fresh RULE)

1. **The sealed draw itself** (draft §4 item 2: `scripts/sealed_mock/draw_sealed_h.py`, the
   commitment, `~/.sealed/…`) — no draw is performed in batch 2; the script is DESIGN only.
2. **Stage (m2)** in full: the pool option (P1/P2), the invariance gate, N_tasks (a cluster
   read of `run_20260729_seed61000` / `sacct 6090909–6090912`), the 55-node evaluate cost
   (94 core-h/venue anchor, row #285) and its scheduling against the 2026-09-23 workspace
   expiry (docket 12b) — addendum A-M3 is Approved in principle but gated on d-sealed-register
   AND "a scheduling word", neither of which batch 2 can supply.
3. **The unseal ceremony** (draft §4 item 4) and the (m2) disposition table (draft §9 rows 3–6).
4. **P-B** (two-sided prior) and the downward h-grid decoupling it needs — a [PHYSICS] build
   node + author word.
5. **The 0.77 pool** (`run_20260729_seed65000_h0p77`, no run_metadata) as a REPORTED-ONLY
   replicate — its provenance recovery is a separate read, not launched.
6. **The design-gate panel record** (§0 blocker): batch 2 can RUN it (zero compute, sonnet ×2)
   but the launch stays blocked until it is GREEN — the STANDING's own wording.
7. Any HALT consequence of a TUNED (m1) read — "HALT paper claims" is the draft's proposed
   consequence and is a fresh RULE on the author's desk, never chair-executed.

---

## 4. Launch blockers for stage (m1), in order (mechanical, for the ops agent)

| # | blocker | how it clears |
|---|---|---|
| 1 | design-gate record ABSENT (STANDING requires a green gate) | chair runs the addendum §1.8 sonnet panel on `REGISTRATION_DRAFT.md` + this dossier; record under `exec/r-sealed-mock/DESIGN_GATE_20260904.md`; GREEN |
| 2 | 0.67 CRB md5 UNKNOWN; `prepared_cramer_rao_bounds.csv` existence UNVERIFIED (cluster unreachable when the draft was written) | the `ssh … md5sum` line of §2; fill §5; pass `EXPECTED_CRB67_MD5` on `--export` |
| 3 | pool link of the 0.67 run dir: does `simulations/injections` resolve to `injection_pool_depth15_50k`? | the same ssh line prints `readlink -f`; a different pool ⇒ STOP and report |
| 4 | `/cluster` preflight READY (was READY ✓ 2026-09-03 with the standing 76-unregistered WARN) | re-run before submission |
| 5 | out-root absent | `test ! -e $WS/graph1_sealed_m1_iiib_20260904` |
| 6 | the sbatch file is uncommitted (batch-2 convention: rsync to `~/darksiren-emri/cluster/`) | rsync, then submit |

Submit line (after 1–6):
```
ssh bwunicluster 'cd ~/darksiren-emri && source cluster/modules.sh && \
  sbatch --export=ALL,RUN_DIR=$WORKSPACE/graph1_sealed_m1_iiib_20260904,VENUE=iiib,EXPECTED_CRB67_MD5=<md5 from §5> \
  cluster/graph1_sealed_m1_headstack.sbatch'
```
joint_r1 (ONLY if the iiib `sacct` sum of Elapsed × 16 ≤ 60 core-h):
```
  sbatch --export=ALL,RUN_DIR=$WORKSPACE/graph1_sealed_m1_joint_r1_20260904,VENUE=joint_r1,EXPECTED_CRB67_MD5=<md5> \
  --time=01:30:00 cluster/graph1_sealed_m1_headstack.sbatch
```
Retrieval: `rsync -aL --exclude='**/simulations/injections' bwunicluster:$WS/graph1_sealed_m1_iiib_20260904/
results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/graph1_sealed_m1_iiib_20260904/`
+ md5 manifest; the reader consumes `simulations/diagnostics/event_likelihoods.csv` and
`simulations/posteriors*/h_*.json` (the (m1) truth is public, so the draft's whitelist rule does
not bind here — stated so nobody applies it by reflex).

---

## 5. Pin slots (filled by the ops agent at first touch, never by this dossier's author)

CRB67_MD5 (`prepared_cramer_rao_bounds.csv`) = `8e9253fef42f574c569a04a3e19299ab` · rows = 1345
data rows (`wc -l` 1346 incl. header; expected 1343 + header = 1344 — g-population disclosure,
+2 rows, not a STOP per this dossier's own rule) · `cramer_rao_bounds.csv` (raw) md5 =
`70cba8a3de9a658e8eef8975c9a61283` · `simulations/injections` → **MISMATCH**: `injections` is a
real directory (not a top-level symlink) whose per-file symlinks resolve into
`$WS/injection_pool_mix200k_20260728`, NOT the canonical `$WS/injection_pool_depth15_50k` —
`readlink -f .../injections` (the sbatch's own check) returns the directory's own path (no
single-target dereference), which the sbatch compares against `readlink -f $POOL` and would
correctly STOP on (blocker table item 3, fired). · cluster HEAD at submit = `06a12422` (fast-
forwarded same session; contains `081b1f28` and matches local HEAD) · job ID iiib = **NOT
LAUNCHED** — see SUBMIT_RECORD_s0c_m1.md (blockers 1 design-gate-absent and 3 pool-mismatch both
unresolved) · job ID joint_r1 = not launched (m1 iiib itself not launched, so the ≤60-core-h
condition on the sibling was never reached).

*Pin taken 2026-09-03/04 by the batch-2 cluster-ops submitter session (read-only ssh commands
only; no sbatch submitted for this node). Full command transcript and blocker disposition:
`results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/batch2_cluster_ops/SUBMIT_RECORD_s0c_m1.md`.*

*Stamp: prereg author B, 2026-09-04. Read-only except this file and
`cluster/graph1_sealed_m1_headstack.sbatch`; no cluster command, no draw, no pool, no edit under
`darksiren_emri/`. Every reading above is chair-provisional and veto-able.*

## PIN CORRECTION (chair, 2026-09-04)

Chair ruling: §2/§5's expectation that the 0.67 run used `injection_pool_depth15_50k` (500 files)
was a factual error, not a data problem. **The pin is corrected to the pool the 0.67 run actually
used: `injection_pool_mix200k_20260728` (707 files).** `cluster/graph1_sealed_m1_headstack.sbatch`
has been edited accordingly (pool-expectation lines only — comment header §"DATA THIS JOB READS",
the `POOL=` assignment, and the `POOL_COUNT -ne 500` STOP threshold, now `-ne 707`; every other
line, including the CRB md5 STOP, the catalogue md5 STOP, and the injections-link mismatch STOP,
is byte-identical). This resolves blocker #3 of §4 as a corrected registration, not as a run
disqualification. Blocker #1 (design-gate record) is separately resolved GREEN by
`exec/r-sealed-mock/DESIGN_GATE_computability_m1.md`.

Re-measured at this pin correction (batch-2 cluster-ops submitter session): `injections` dir file
count 707; `injection_pool_mix200k_20260728` file count 707 (match); pool file-list md5
`a1dffdf561c51c8c778dce115c5fb371` (no manifest file exists in the pool dir); CRB67 md5 and row
count unchanged from the §5 pin (`8e9253fef42f574c569a04a3e19299ab`, 1345 data rows). Full
transcript: `PIN_RECORD.md` (appended) and
`exec/batch2_cluster_ops/SUBMIT_RECORD_m1.md`.
