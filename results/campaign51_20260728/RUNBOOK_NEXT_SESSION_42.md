# Runbook 42 — graph-1 mid-batch handoff for the Fable 5.1 session (supersedes runbook 41)

**Read first.** The 2026-09-01→03 session executed Research Graph 1 waves 1–2 end-to-end
(rows #291–#330). The session restarts NOW (model upgrade) with TWO measurements in flight and
watchers that DIED with the old session — §1 is the very first thing to do.

## 1. FIRST ACTIONS — re-arm the watchers (nothing else before this)

1. **m-s3 cell S (LOCAL process, survives the restart):** PID in
   `graph1_20260901/exec/m-s3-postflip-coverage/pidS... ` — actually inv-1 PID file is
   `pids_inv1.txt` (first number = cell S, PID 2428302), log `cellS_inv1.log`, work root
   `tree2_20260830/b8_cal_harness_work_s4_postflip/`. At handoff: ~60/100 universes (n_U_min=60
   floor CROSSED → read-valid whenever it stops). Its 24h `--max-wall-s` expires ~2026-09-03
   10:55 CEST → it will stop `wall_limited` and write `_run_status_S.json`. THE FROZEN RULE
   (r-b82-s4 §3, ratified row #301 item 2): resume-to-complete allowed, ≤3 invocations —
   **invocation 2 launch** (from REPO ROOT — the catalogue path is cwd-relative, row #288/#320
   gotcha): `nohup .venv/bin/python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py
   --work-root <ABSOLUTE work root> --N 200 --cell S --seed-block 901000 --n-universes 100
   --max-wall-s 86400 > exec/m-s3-postflip-coverage/cellS_inv2.log 2>&1 &` (checkpoints skip
   automatically). Cell T is COMPLETE (25/25, row #326) — do not touch its work.
2. **S0-B production (CLUSTER, survives):** job **6779532** (array 0-4, 5 registered θ-nodes;
   the first submission 6779448 failed 5/5 at 11s on a strict-equality HEAD pin, fixed to an
   ancestor check and resubmitted — rows #330-#331). Re-arm a sacct watcher. On completion: retrieval + the PA-HIER-33 /
   score_b_re reads (g-score-null |Z|≤3 gate per rows #225/#251/#287) → d-photoz-leverage
   dossier to the author. Cost anchor: 74.7–101.4 CPU-h across 5 tasks (~15–23 CPU-h each).
3. SSH is authenticated as of handoff; `ControlPersist` expires ~8h (OTP, author-only re-auth).

## 2. State of record (rows #291–#330, all committed + pushed to origin, HEAD f82f77a8+)

- **Docket items 1–11 all ruled** (author rows #301/#304/#308/#314; chair-under-grant row #325 —
  the author granted "you can make decisions but flag them", every chair decision is veto-able).
- **SETTLED ×2:** F-ii window (not adopted, rows #302/#312/#314) · q-a4-provisional (falsifier
  INSIDE-BOTH row #322, A4 final row #325).
- **[PHYSICS] landed ×2 this session:** a26959b4 (h_grid_admissibility_max decoupling; byte-id
  GREEN row #311; G-EXT 55-node grid complete row #313) · 2b657255 (Option A′ class-G
  S̄_φ de-double-weight; fleet ran on it).
- **m-jr1 Z-CONFIRMED** (zero-compute registered read, row #305; claim promotion at d-calibration).
- **Interactive docket artifact:** https://claude.ai/code/artifact/4319714c-7ff7-4166-a86d-329338c01df1
  (dark-first; update via url param from the new session). Research-interests essay: artifact
  538dd463 (non-record).

## 3. OPEN AUTHOR WORDS (docket item 12, rows #328): 12a backup destination for the 159GB
sole-copy ~/emri-archive (TOP PRIORITY) · 12b cluster evacuation before Sep 23 (~250GB,
0 extensions) · 12c disk culls · 12d DATA_INVENTORY commit · 12e merge→main (112 ahead) ·
12f safe builds (renderer/readout/utilities now; EvaluationConfig after merge) · 12g docs sync.
Health scan: `graph1_20260901/health_scan/` (synthesis + 4 lenses + archive-fix record).

## 4. Then-next queue (graph order)

1. Cell S concludes → aggregation (`--score-only --population 200` per cell +
   `--score-only-ratio-t-s`, from repo root) → **rd-s3-readout** (three-valued existence;
   the two design-gate caveats routed there: general-n exact-KS computed externally; the
   binom_bands normal-approx-vs-"exact" label, row #303) → F value → **r-completion-residual +
   r-cone-loss authoring** (ONE top-tier prereg author, k-wave2 cap) → d-completion/d-cone-register
   [RULEs] → the measures → **d-calibration** dossier (needs: rd-s3-readout green, rebaseline
   banked-as-comparand ask, m-jr1 disposition) → d-residual-attribution → the three paper rulings.
2. S0-B reads → **d-photoz-leverage** [RULE].
3. GATE-ACC addendum (reporting-only) may still be computing on the login node — check
   `exec/v-falsifier-ii-classG/` notes.

## 5. Standing gotchas (new this session, on top of runbook 41 §4's)

- Comparand FLAG-STATE pinning (rows #298/#299: a flag-mismatched comparand manufactures
  mechanistic-looking phantom deltas) · byte-id gates are SAME-MACHINE by definition (row #325
  decision; cross-machine FP 1e-16→1e-9 is not a defect, rows #318/#319) · `&` backgrounds a whole
  `&&`-chain · `source X | tail` subshell eats env vars · double-quoted ssh remote strings expand
  $vars LOCALLY (row #317) · run-dir symlinks to shared injection pools must be excluded from
  `rsync -aL` (row #311) · untracked-scp'd-then-committed sbatch files collide with cluster pulls
  (move aside, row #330) · the b8 harness must run from REPO ROOT · local OOM with 3 heavy
  processes on the 30G box (row #321) — stagger heavy local runs.
- Ledger clerk convention: one long-lived clerk agent wrote rows #291–#330 (quote-verbatim,
  chair-derived marked); re-create one in the new session.

## 6. Operating mode (unchanged, rows #290/#325)

Orchestrator = chair: delegates writes/runs (sonnet for mechanical, top-tier only for
derivation/prereg/decisive verification, ≤4 identities per batch), reviews, adjudicates, commits.
Gate-panel law: no science read without green-or-waived stamps. Every event = a ledger row.
The author's autonomy grant (row #325) stands: decide-and-flag, veto reverts.
