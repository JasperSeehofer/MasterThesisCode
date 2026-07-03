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

- ☑ **[HIGH — issue #19] stale d_L pre-screen FIXED** (2026-07-02, `/physics-change`,
  user-approved): `luminosity_distance_prescreen_gpc(z_max, h)` = 1.05 × d_L(rate-model
  z_max; runtime h), computed once per run; WARNING on hit. Worse than reported: the old
  2.0 Gpc cutoff was already inside the z ≤ 0.5 host-draw volume (d_L(0.5)=2.74 Gpc).
  ☐ remaining: re-measure `PRESCREEN_DL_MARGIN` (placeholder 1.05) on post-dt² injection
  data — same smoke run as the SNR-factor check below; issue #19 stays open for this.
- ☐ **[DESIGN — decide before submit] `HOST_DRAW_Z_MAX = 0.5` is horizon-stale too.**
  Its justification comment ("detection horizon z ≈ 0.18, so z < 0.5 is safely beyond →
  truncation EXACT, p_det = 0 beyond") is pre-dt² reasoning; at physical SNR, p_det > 0
  well beyond z = 0.5, so the z-cut now truncates *detectable* population. The sign-off
  expects the campaign population to deepen to z ≲ 1.5 — that requires raising
  HOST_DRAW_Z_MAX with GLADE completeness machinery beyond z = 0.5 (becomes binding) and
  `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT = 0.55` alongside. Sim+inference use the cut
  consistently, so this is a population-scope *decision*, not a silent bug — but the
  campaign's science reach (event yield, forecast depth) hinges on it. Needs its own
  issue + user decision + `/physics-change`.
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

## 4b. PRE-REGISTERED "bias resolved" criterion (fixed 2026-07-03, BEFORE submission)

Defined before any campaign data exists, per the referee's asserted-decisiveness
critique (report either outcome):

1. **Multi-seed accuracy (primary)**: over the 4 seeds at h_true = 0.73, the
   per-seed volume_deconv 1D MAP sample must satisfy
   |mean(MAP) − 0.73| < 2·SEM (SEM = std(MAP)/√4). Same check for posterior means.
2. **Closure**: each closure run (0.67, 0.77) recovers its truth inside its own
   68% HPD interval (both channels).
3. **Calibration**: per-seed `pp_coverage` (G4b, volume kernel) at the campaign's
   photometric width stays near-nominal — cov68 within ±0.10 of 0.68 at n=250
   realizations (binomial σ≈0.03; the 2026-07-03 σ_z/z scan bounds validity at
   σ_z/z ≲ 0.8).
4. **Verdict language**: all three pass → "bias resolved at the campaign's
   statistical precision"; any fail → report the measured residual with its
   multi-seed uncertainty — no stronger claim.

## 5. Paper hooks

- Paper A (`paper_a/`, drafting in flight): consumes the seed600 confirmation (§0) into
  its RESULT PENDING slot; campaign P–P-across-seeds figure is an optional strengthener.
- Paper B: campaign is its data source; wire the one-command `paper/figures` regeneration
  script during harvest, not after.
