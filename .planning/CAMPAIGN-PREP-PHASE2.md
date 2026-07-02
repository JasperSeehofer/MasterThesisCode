# Phase-2 Multi-Seed Production Campaign — Prep Runbook (2026-07-02)

Derived from `.planning/gate/GATE_SIGNOFF.md` (gate PASSED, campaign unlocked) and the
approved publication roadmap. This is the single checklist to run down before and during
the campaign. Owner: Jasper. Status legend: ☐ open · ☑ done · ⏳ blocked-on.

## 0. State at prep time

- ☑ Gate G1–G11 signed off; PR #18 **merged to main** 2026-07-02 16:45Z (merge incl. the
  docs-CI docstring fix `e283354` + G4b CHANGELOG/CLAUDE.md entries).
- ⏳ seed600 full-grid confirmation (eval `5698617` + combine `5698618`) still queued;
  partition widened in place to `cpu,cpu_il` (no cancel — priority-age preserved;
  `cpu_il` jupyter reservation lifts weekdays 20:00). Retrieval recipe:
  `.planning/HANDOFF-DERAIL-CLUSTER-CONFIRM-20260702.md` §"When the combine COMPLETES".
  Expected: peaked ~0.73. Paper A slot marked RESULT PENDING.

## 1. Blockers found during prep (fix BEFORE submitting the campaign)

- ☐ **[HIGH — issue #19] `LUMINOSITY_DISTANCE_PRESCREEN_GPC = 2.0` is stale.**
  Calibrated on retired pre-dt² injection data ("no detectable EMRI beyond 1.66 Gpc");
  at physical SNR the horizon is z ≲ 1.5 (d_L ~ 11 Gpc). Running the campaign with it
  silently truncates the detectable population (DEBUG-level log only).
  Fix via `/physics-change`: derive from the M1 rate model's z_max at the smallest
  grid h (× margin), or minimally bump to ≳ 16 Gpc and re-measure from the first
  post-fix injection pool. **Do not submit the campaign before this lands.**
- ☐ **`PRE_SCREEN_SNR_FACTOR = 0.3` empirical re-check** (sign-off campaign-time action).
  Ratio-based, survives the dt² rescaling in principle, but the false-negative rate must
  be re-measured at the new depth (longer, more-redshifted waveforms): in the smoke run,
  disable the quick-SNR gate for N≈100 events, record (quick_snr, full_snr) pairs, verify
  no full-SNR ≥ 20 event has quick < 6. Same `/physics-change` run as issue #19 if the
  factor moves.

## 2. Blocked on seed600 jobs finishing (do NOT touch the cluster repo before)

The queued eval must run against the cluster state it was configured for
(branch @ `6d4c4e1`, z_helio catalogue) to stay apples-to-apples with the archived
railed baseline. After combine `5698618` completes and results are retrieved+verified:

- ☐ `git fetch && git checkout main && git pull` on `~/MasterThesisCode` (cluster).
- ☐ Stage the **rebuilt z_cmb catalogue** (local rebuild 2026-07-02; 99.9 % rows shifted,
  median |Δz| 6e-4):
  `rsync -avz --partial master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv bwunicluster:MasterThesisCode/master_thesis_code/galaxy_catalogue/`
  then re-run preflight → require `VERDICT: READY ✓` **and 8-col schema confirmation**.
- ☐ Note: committed de-rail/ablation baselines refer to the z_helio catalogue (sign-off
  "Open, non-blocking"); the campaign runs z_cmb — never mix in one comparison.

## 3. Campaign design (per roadmap Phase 2)

- Injections: regenerate at **single h_ref = 0.73** with the M_z convention
  (p_det h-invariance established 2026-06-21; simplifies `submit_injection.sh` — see
  memory `project_injection_todo`). Post-fix SNR scale ⇒ deeper pool: budget for more
  events per task and re-tuned `--time` (see §4 timeouts).
- Simulations: **4 seeds @ h_true = 0.73** + closure truths **0.67 / 0.77** (1 seed each).
- Inference: `--normalization_mode volume_deconv` (default), seeded (`--seed` reaches
  inference, G4); 38-value h-grid first pass.
- Budget ~1800 cpu-h + GPU sim time. Queue strategy: `cpu` + `cpu_il` multi-partition
  (this prep's escalation shows `cpu_il` drains overnight); submit before 20:00 only to
  `cpu`; **never cancel to chase slots** (W-LOOP-04).
- Workspace: `ws_find emri` → extended to 2026-08-31 (Phase 0). `ws_extend emri 60` again
  if the campaign slips past mid-August. Immediate rsync-back after every stage.
- Provenance: `run_metadata_<task>.json` carries git_commit+seed+args; tag the campaign
  base commit (`campaign-phase2-base`) before first submit; DATA_INVENTORY tier update
  with the new datasets; retire/label pre-dt² datasets explicitly.

## 4. Campaign-time actions (from the sign-off — wire these into the runs)

- ☐ **Timeout histograms by (M, e₀, p₀)**: G9 landed per-parameter logging; harvest the
  logs of the smoke run + first seed into histograms (0.6–1.25 %/stage at the old scale;
  expect drift at longer waveforms). If loss concentrates in a corner, quote it in the
  Paper-B systematics rather than raising timeouts blindly.
- ☐ **κ-gate exclusion count**: G10 gate (κ > 1e14 skips event) — count skips per seed
  from logs; report alongside the timeout budget.
- ☐ **Per-seed `pp_coverage` runs**: `master_thesis_code.validation.pp_coverage` (G4b)
  once per seed; collect into one P–P figure across seeds.
- ☐ **Finer-grid pass** on the final combined posterior (superdense grid around the MAP;
  posterior filenames support 4-decimal h since b1933d8).
- ☐ **Smoke test first**: `--tasks 2 --steps 10` (cluster skill golden rule) + the §1
  pre-screen measurements, THEN full submission.

## 5. Paper hooks

- Paper A (`paper_a/`, drafting in flight): consumes the seed600 confirmation (§0) into
  its RESULT PENDING slot; campaign P–P-across-seeds figure is an optional strengthener.
- Paper B: campaign is its data source; wire the one-command `paper/figures` regeneration
  script during harvest, not after.
