# DESIGN_GATE_computability_m1 — fresh computability-only design gate, stage (m1)

Node: m-closure067-headstack (r-sealed-mock, stage m1). Reviewer: fresh sonnet
computability-only design-gate reader (batch 2, this is the record dossier §0/§4
blocker 1 asks for — `exec/r-sealed-mock/DESIGN_GATE_20260904.md` naming in the
dossier; filed here under the requested name `DESIGN_GATE_computability_m1.md`).
Scope: computability only. No science read performed, no aggregate computed over
any registered population, no file under `darksiren_emri/` edited, no
`INFORMATION_FORECAST*` file opened. Inputs read: `REGISTRATION_DRAFT.md`,
`D_SEALED_REGISTER_DOSSIER.md`, `cluster/graph1_sealed_m1_headstack.sbatch`, plus
(for the flag-by-flag and anchor cross-checks the brief asks for)
`cluster/graph1_headrebaseline_iiib.sbatch`, `cluster/graph1_headrebaseline_joint_r1.sbatch`,
`darksiren_emri/validation/correspondence_1d.py` (constants only — `PRODUCTION_FLAGS`,
`REGISTERED_RESOLVED_FLAGS`, `H_GRID_41` — read, not executed), and
`exec/m-head-rebaseline/READOUT_RECORD.md` (row #302 anchor cross-check).

**Verdict: GREEN.** No defect found that makes the (m1) run wrong or unregistered.
Two non-blocking notes are recorded in §6 for the record.

---

## 1. CLI verbatim check

`cluster/graph1_sealed_m1_headstack.sbatch`'s python invocation (lines 197–225) was
diffed token-by-token against `cluster/graph1_headrebaseline_iiib.sbatch` (lines
120–149) and `cluster/graph1_headrebaseline_joint_r1.sbatch` (the joint_r1 delta).
All 27 flags/values are identical, in identical order:
`--evaluate --h_value --seed --simulation_index --strategy physics-floor
--pdet_dl_bins 60 --pdet_mass_bins 40 --pdet_estimator local_linear
--pdet_z_resolved --fisher_cond_threshold 1e16 --host_z_kernel volume_deconv
--host_mass_kernel auto --normalization_mode absolute_marginal
--selection_in_completion_numerator fused --catalogue_mass_overlap production
--catalogue_mass_error_scale 1.0 --completion_b_scale derived --eddington_m on
--sigma4d_mass_kernel point --completion_event_measure ratio
--catalogue_global_selection phi --mass_filter_geometry linear --mass_filter_k 1.5
--theta_b 0.0 --theta_s 1.0 --theta_sites all [OBS_ARGS] --log_level INFO`. The
`${OBS_ARGS[@]+"${OBS_ARGS[@]}"}` insertion sits at the exact same position the
joint_r1 sibling puts its literal `--observed_catalogue "$OBS_CATALOGUE"`, and is
the standard `set -u`-safe empty-array idiom (no expansion when `VENUE=iiib`).

Cross-checked the literal values against `PRODUCTION_FLAGS` in
`correspondence_1d.py:330-339` (normalization_mode, host_z_kernel,
selection_in_completion_numerator, catalogue_mass_overlap, completion_b_scale,
pdet_dl_bins, pdet_mass_bins, pdet_estimator) — all 8 match the sbatch literals
exactly. Cross-checked the remaining CLI values against the overlapping keys of
`REGISTERED_RESOLVED_FLAGS` (`correspondence_1d.py:3154-3168`, 13 keys, matching
the dossier's "13 tokens"): `catalogue_global_selection=phi`,
`mass_filter_geometry=linear`, `mass_filter_k=1.5`, `theta_b=0.0`, `theta_s=1.0`,
`theta_sites=all` all match. The remaining 5 `REGISTERED_RESOLVED_FLAGS` keys not
directly settable by a CLI flag in this list (`catalogue_numerator_survival`,
`catalogue_numerator_survival_2d`, `mass_filter_sigma`, `theta_phi_divisor`,
`theta_zwindow`) are left to resolve at their code default, per the documented,
deliberate blindness convention shared with `graph1_headrebaseline_iiib.sbatch`
(no `--catalogue_numerator_survival_2d*` flag passed; `catalogue_leg_1d_mass_aware`
not passed → resolves via its own `auto` default) — consistent, not a gap.

**GREEN.**

## 2. Array shape, seeding, cores, freeze mechanics

- `H_VALUES` (sbatch lines 165) is character-for-character identical to
  `H_GRID_41` in `correspondence_1d.py:353-358` and to the array in
  `graph1_headrebaseline_iiib.sbatch` — 41 entries, `--array=0-40` matches
  (`#SBATCH --array=0-40`); index 21 = 0.730 confirmed by direct count.
- Seeds: `EVAL_SEED=777000; TASK_SEED=$((EVAL_SEED + TID))` — matches "777000+TID"
  and is byte-identical to the head-rebaseline convention (gotcha 4).
- Cores: `#SBATCH --cpus-per-task=16` — matches the head-rebaseline sibling and
  `graph1_headrebaseline_iiib.sbatch:44`.
- Ancestor pin: `git merge-base --is-ancestor 081b1f28 HEAD` (STOP if not), plus
  `git diff --quiet 081b1f28 HEAD -- darksiren_emri/` (STOP if dirty) — both
  present (lines 75-85). `081b1f28` is a real commit in this repo's history
  (`fix(cluster): s0b sbatch HEAD pin — ancestor check instead of strict
  equality`, visible in `git log`). The registration draft's demand for
  *strict*-equality freeze (§4 item 1) is explicitly scoped by the dossier to the
  future sealed (m2) run only — "for (m1) this is the 'code freeze' of draft §4
  item 1 in its ancestor form; the STRICT-equality form is reserved for the
  sealed (m2) run" (dossier §2) — a correctly disclosed, non-contradictory
  scoping, since (m1)'s truth (0.67) is public and has no seal to protect.
- `EXPECTED_CRB67_MD5`: required via `: "${EXPECTED_CRB67_MD5:?...}"` (hard STOP
  if unset on `--export`), then checked against the computed md5 of the staged
  `prepared_cramer_rao_bounds.csv` (lines 65, 122-126) — present and correctly
  gates before the array proceeds.
- Injection-pool link check: `POOL_COUNT` must equal 500 (STOP otherwise, lines
  107-111) and, if the 0.67 run's own `simulations/injections` symlink exists, it
  must resolve to the same canonical pool (STOP on mismatch, lines 112-118) —
  present, matches dossier §2's registered procedure.
- `joint_r1` conditional switch: sha256 pin `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`
  matches `graph1_headrebaseline_joint_r1.sbatch` verbatim; the sidecar
  `parent_csv` resolution check (gotcha 10) is additionally inlined into this
  script (the joint_r1 sibling relies on a pre-flight check instead) — a strictly
  additional safety check, not a divergence from the registered CLI. `VENUE` is
  validated (STOP on any value other than `iiib`/`joint_r1`).

**GREEN** on all of: 41-task `H_GRID_41` array; seeds 777000+TID; 16 cores;
ancestor pin; `darksiren_emri/` diff-quiet check; `EXPECTED_CRB67_MD5 --export`
STOP; the injection-pool link check (500 files); the joint_r1 conditional switch.

## 3. Reading rule

The three read-bearing cells for (m1) — TUNED, NOT-TUNED (subsuming
NOT-TUNED-AT-RAIL as a disclosed bound, not a fourth independent value), and
INTERMEDIATE — plus the non-reading NO-READ gate-failure state, are defined in
`D_SEALED_REGISTER_DOSSIER.md` §2 with fixed numeric thresholds:
`|mean_h − 0.6659| ≤ 3σ_h` and `|MAP − 0.665| ≤ 3σ_h` (2D); `|mean_h − 0.6670| ≤
3σ_h` (1D). These anchors were cross-checked directly against
`exec/m-head-rebaseline/READOUT_RECORD.md:39-40`: 2D `mean_h=0.665854,
sigma_h=0.018475, MAP=0.665`; 1D `mean_h=0.666987, sigma_h=0.017526, MAP=0.665` —
**exact match**, correctly sourced from row #302. σ_h used in the live bands is
the (m1) run's own measured σ_h at N=1343 (draft's disclosed ≈1.09× scaling from
the N=1588 anchor), not a re-use of the anchor value itself — a mechanical,
zero-fresh-choice computation from the run's own output, not a fresh statistical
decision.

**GREEN.**

## 4. Leak inventory / binary-read independence from the leaked h

The 0.67 truth is disclosed as already public by construction (directory name,
every `run_metadata_*.json`, prior readouts quoted in draft §0/§2 and dossier
§1.1) — (m1) is explicitly registered as NOT the T-1 sealed verdict, and the
dossier states in terms that leave no ambiguity that the draft's blindness
whitelist (`RUN_DIR/logs`, `run_metadata_*`, the CRB, the sealed file off-limits
between draw and unseal) "does NOT apply to (m1)... stated so nobody applies it
by reflex" (dossier §1.1, §2 retrieval note). Separately and more importantly for
this check: the (m1) cell-bearing criteria themselves (§3 above) are all fixed
numeric constants (0.6659, 0.665, 0.6670, 3σ_h) that do not reference h_inj/0.67
anywhere in their formulas — the only place 0.67 enters the (m1) documents is
the REPORTED-ONLY, non-cell-bearing comparison `|mean_h − 0.67|` (dossier §2,
"reported alongside... not cell-bearing"). So the binary TUNED/NOT-TUNED read is
structurally independent of the leaked truth value, and the one place the truth
is used is explicitly marked non-decisive.

**GREEN.**

## 5. Cost vs cap

Draft §7 [LOCAL]: 41 tasks × 16 cores × (5.57–6.60 min/task, from the retrieved
`graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/` log-name →
`.out`-mtime deltas, N=1588) = 61–72 core-h; scaled by 1343/1588 (the (m1) pool's
event count) if runtime is linear in N ⇒ 52–72 core-h. Both the lower bound
(scaling assumed linear in N: 61×0.846≈52, 72×0.846≈61) and the conservative
"no-benefit-from-fewer-events" reading (the unscaled 61–72 itself, if per-task
wall time is dominated by fixed overhead rather than N) fall inside the
registered range and, either way, under the **75 core-h/venue cap** (dossier §2,
row #349 wording). The timing source is named and independently locatable (the
retrieved re-baseline run directory's own logs). The joint_r1 sibling is gated
separately, outside the sbatch, on a measured `sacct` cost of the iiib array (≤
60 core-h) before it is submitted — an explicit, mechanical, ops-agent-executed
condition (dossier §1 item 6, §4), not baked as an automatic branch in the
script (appropriately so, since it depends on data the script cannot see before
the first array finishes).

**GREEN** — cost is bounded under cap either way; see §6 note (a) for a
presentation nit in how the 52–72 range is written.

## 6. Zero fresh choices; kill criterion / max_revisions / blindness line

- **Zero fresh choices:** every value the sbatch consumes is either hardcoded
  from a cited source (H_GRID_41, the CLI flags, the two md5/sha256 pins, the
  ancestor commit) or taken as a pin-at-first-touch procedural read
  (`EXPECTED_CRB67_MD5`) whose *procedure* — not its numeric outcome — is
  registered in advance; no branch in the script embeds a choice that was not
  already fixed by the registration text. Confirmed by direct read of the script.
- **max_revisions:** `GRAPH1_ADDENDUM_PROPOSAL_20260903.md` §1.7 states
  "r-sealed-mock max_revisions 1 (a sealed test cannot be re-drawn without
  destroying the seal)" for the r-sealed-mock node as a whole; the draft's own
  §4 item 5 reconciles this against a chair launch note of 2, registering
  "design revisions of this un-drawn registration ≤ 2... sealed draw ≤ 1",
  ratified-and-flagged by the dossier (§1 item 3). (m1) itself is not the sealed
  draw (it consumes only pre-existing, public data) and the sbatch is
  idempotent by construction — a resubmission skips any h-node whose posterior
  JSONs already exist (lines 176-183) — so it carries no informational cost from
  re-running and needs no cap of its own distinct from the node-level one
  already registered. This is present and adequate, not a gap.
- **Kill criterion:** `GRAPH1_ADDENDUM_PROPOSAL_20260903.md` line 60 states
  kill_criterion is "mandatory per infra 2.1" for **question (q-)** nodes
  specifically; r-sealed-mock is a register node and m-closure067-headstack/(m1)
  a measure node, whose governing-column equivalent is the gate/consequence
  text, not a q-node kill_criterion. (m1)'s functional equivalent is present:
  the NO-READ gate-failure state ("nothing banked" — dossier §2 table) plus the
  per-outcome disposition table (draft §9: TUNED → HALT paper claims proposed to
  the author, fresh RULE; NOT-TUNED → the anti-tuning stamp stays PARTIAL,
  fresh RULE by the document's own header convention, draft line 6-7 "every...
  return as fresh RULE"). No literal "kill criterion" label is required or
  missing relative to the governing rule.
- **Blindness line:** present and explicit — dossier §1 item 8 and §1.1 state in
  terms that leave no room for reflexive misapplication that the sealed-draw
  blindness protocol does not bind (m1), because (m1)'s truth is public.

**GREEN**, with two non-blocking notes for the record (not defects, not
gating):

(a) The 52–72 core-h cost line in draft §7 mixes an N-scaled lower bound (52 ≈
61×1343/1588) with an un-scaled upper bound (72, i.e., not further scaled down
by the same 1343/1588 factor). Read charitably this is the intended envelope —
52 if wall time scales with event count, 72 if it is dominated by fixed
per-task overhead and does not — but the draft does not say so explicitly. Since
both readings sit under the 75 core-h/venue cap, this has no bearing on
launchability; a one-line clarification in the draft would remove the ambiguity
for a future reader.

(b) The design-gate record this file provides discharges dossier §0/§4 blocker
1 ("design-gate record: ABSENT... the (m1) sbatch is written and ready but is
NOT launchable under the STANDING until a design gate is run and GREEN"). This
review is computability-only per its own charter; it does not touch, and is not
a substitute for, any scientific-content review of the (m1) registration (the
prior, band-content and disposition-table judgment calls are the author's, per
CLAUDE.md).

---

## 7. Summary table

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | CLI verbatim = production CoR-P set / `REGISTERED_RESOLVED_FLAGS`; array/seed/cores/ancestor-pin/diff-quiet/md5-STOP/pool-link/joint_r1 switch | GREEN | §1–2 above |
| 2 | reading rule three-valued, thresholds from row #302 (σ_h 0.018475, mean 0.665854) | GREEN | §3 above; `READOUT_RECORD.md:39-40` |
| 3 | leak inventory stated; binary read independent of leaked h | GREEN | §4 above |
| 4 | cost 52–72 core-h ≤ cap 75, timing source named | GREEN (note a) | §5 above |
| 5 | zero fresh choices; kill criterion/max_revisions/blindness line | GREEN | §6 above |

**Overall: GREEN.** Stage (m1) is launchable, subject to the mechanical blockers
already listed in `D_SEALED_REGISTER_DOSSIER.md` §4 (items 2–6: md5 pin at first
touch, pool-link verification, `/cluster` preflight, out-root-absent check,
committing/syncing the sbatch) — none of which are computability defects, all of
which are already registered as pre-submission procedure.

*Reviewer: fresh sonnet computability-only design-gate reader, 2026-09-04
(matching the dossier's dated blocker). No cluster command run, no science
computation performed, no aggregate computed over any registered population, no
edit under `darksiren_emri/`, no `INFORMATION_FORECAST*` file opened.*
