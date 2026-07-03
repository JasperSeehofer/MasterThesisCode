# Handoff — Phase-2 campaign submission session (2026-07-03)

Fresh-session entry point for the multi-seed production campaign. Companion to the
runbook `.planning/CAMPAIGN-PREP-PHASE2.md` (on `campaign/phase2-prep`, → main once
PR #21 merges); this file adds the two user decisions of 2026-07-03, the measured
walltime anchors, and the exact ordering. Base yourself on main after PR #21.

## 0. Pre-state (verify, don't assume)

- Paper A: peer-review R1 = major_revision (report `GPD/publication/paper-a-main-7a640e3609ef/REFEREE-REPORT.md`);
  the seed600 full-grid confirmation is DONE and filled (commit `2519dc3` on `paper/paper-a-draft`:
  1D MAP 0.745, edge mass <1e-75; artifacts `results/commission_20260701/redteam/combined_posterior_voldeconv_fullgrid{,_with_bh_mass}.json`).
- Cluster: no jobs in queue; repo parked at `physics/derail-completion-4pi` @ `6d4c4e1` —
  the baseline-consistency hold is LIFTED; workspace `emri` valid to 2026-08-31.
- ☐ **PR #21 merged?** (`gh pr view 21`) — CI green 2026-07-03; if still open, user merges first.

## 1. USER DECISIONS (2026-07-03, recorded in this session; comment/close on the issues when landing)

1. **Issue #20 — `HOST_DRAW_Z_MAX = 1.5`** (full rate-model depth, per gate sign-off).
   Raise `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT` (0.55 → ~1.55) alongside.
2. **Issue #16 — value-correct host peculiar velocities NOW (option a)** — chosen precisely
   because it is re-simulate tier (post-campaign adoption would retire the data). Use the
   GLADE+-provided PV-corrected redshifts where flagged (issue cites flag col 29 / error
   col 30 in raw GLADE+ — VERIFY indices against `galaxy_catalogue/handler.py` and
   `.planning/derivation-photoz-incatalog/FRAME-SYSTEMATIC.md` §3); treat the error column
   as the uncertainty of the *applied* correction (today: error-inflation only,
   `handler.py:302-307`). Residual coherent bulk-flow stays a budget row.

Both fold into ONE catalogue rebuild + ONE `/physics-change` set on a fresh branch off main
(suggest `physics/campaign-depth-pv`): constants + rebuild GLADE+ reduction (depth ≤~1.55,
z_cmb frame, PV-corrected z, 8-col schema) + completeness-machinery validation at z > 0.5
(per-pixel Schechter m_th machinery becomes BINDING — GLADE+ completeness < 0.5 out there;
that is the catalogue-dominated regime Paper A claims, so this is a feature, but validate).
GLADE+.txt exists ONLY on the dev box (cluster cannot rebuild). Trigger files ⇒ `/physics-change`
hard gate; `[PHYSICS]` commits; close #20 + #16 with commit refs; #19 stays open for §4.

## 2. Final readiness sweep (AFTER §1 lands — it changes the constants being audited)

One multi-agent sweep (Workflow) over the known pitfall classes; propose-only, fix-then-verify:
- horizon-stale constants (the #19/#20 class): grep `constants.py` for anything derived from
  pre-dt² horizons/scales; check `PRESCREEN_DL_MARGIN` (placeholder 1.05), `PRE_SCREEN_SNR_FACTOR=0.3`.
- grid coverage: p_det injection grid must span the NEW d_L/M_z ranges (z ≲ 1.5); H_VALUES grids;
  prepared-CRB filters at the new SNR scale.
- timeout/walltime budgets in every sbatch (see §5 anchors) + waveform timeouts (30s/90s) at longer waveforms.
- catalogue: 8-col schema, z_cmb + PV provenance stamped, staging plan (1.6+ GB rsync).
- guards: archive/idempotency two-guard pattern intact; seed convention BASE_SEED+TASK_ID; run_metadata provenance.
- M_z convention wired at every injection/consumption site (W-PRE-12 class: check EVERY output writer).

## 3. Cluster sync (after §1–§2)

`git checkout main && git pull` on cluster; rsync the NEW reduced catalogue; re-run
`cluster/preflight.sh` → require `VERDICT: READY ✓` **and** 8-col confirmation.
Never mix z_helio-baseline comparisons with z_cmb-campaign results.

## 4. Smoke test FIRST (`--tasks 2 --steps 10`)

Measures at the new depth: per-stage throughput (→ §5 sizing), `PRESCREEN_DL_MARGIN`
re-measurement (then close #19), `PRE_SCREEN_SNR_FACTOR` false-negative check (disable
quick gate for N≈100 events; no full-SNR ≥ 20 with quick < 6), first G9 timeout-histogram
sample, κ-gate skip count.

## 5. Walltime sizing (2026-07-03 measured anchors — never reuse budgets across a scale change)

| stage | anchor | rule |
|---|---|---|
| simulate (GPU) | none valid post-dt² | size from smoke, ≥2× measured |
| injections (GPU) | old 72k pool is pre-dt² depth — regenerate, single h_ref=0.73, M_z convention | size from smoke, ≥2× |
| evaluate (CPU/h-value) | **56–76 min @ 3355 ev / 16 cpus** (volume_deconv, jobs 5732036) — the shipped `evaluate.sbatch` 15-min/128-cpu default and the first 25-min guess were both wrong | linear in events: budget 6h @16 cpus at 2–4× yield, or 64 cpus; confirm from smoke |
| combine (CPU) | 45 min DIED in figure phase (job 5735965); posteriors alone ≈ 20 min | 90 min AND skip cluster-side figures (make them locally) |
| any array | per-node variance is real (one task needed 2× on uc2n771) | `resubmit_failed.sh` for stragglers; never cancel-to-chase (W-LOOP-04) |

## 6. Submit (design per runbook §3)

Tag `campaign-phase2-base` first. 4 seeds @ h_true=0.73 + closure 0.67/0.77 (1 seed each);
inference `volume_deconv` (default) + `--seed`; 38-value grid first pass. Queue: `cpu` before
20:00 (+`cpu_il` nights/weekends); immediate rsync-back after every stage; DATA_INVENTORY
tier update + explicit retirement labels for pre-dt² sets.

## 7. PARALLEL: synthetic coverage at order-unity σ_z/z (Paper A referee blocker REF-P001/S006)

Local CPU, `master_thesis_code.validation.pp_coverage` (config anchor: committed run used
σ_z=0.035, n_real=250 — σ_z/z ~ 0.1–0.2 at its event redshifts). Raise σ_z until σ_z/z ≈ 0.5–1
at the typical event z (e.g. σ_z ∈ {0.10, 0.15, 0.25}), both kernels, ≥250 realizations,
paired seeds. Verdict = volume-kernel 50/68/90 coverage at order-unity ratio.
**Safe to run while sims queue: simulations are estimator-independent** — a bad coverage
verdict changes inference mode/paper framing only; cancel only eval stages if needed.

## 8. Paper A hooks (do not lose)

- Campaign outcome feeds the revision round: define BEFORE submission what "bias resolved"
  means (multi-seed MAP distribution vs truth at measured scatter; near-nominal coverage at
  photometric width) — report either outcome (referee dinged asserted decisiveness).
- Cheap campaign-independent repairs can interleave: REF-M001 false sentence (realdata.tex:54
  area), Cross-Parkin 2025 + 2025–26 literature rebuild (REF-L001/L002), mechanism attribution
  (REF-P004), proof-redteam follow-ups 1–6 (`GPD/publication/paper-a-main-7a640e3609ef/review/PROOF-REDTEAM.md`).
  HOLD abstract/intro/conclusions regime framing until campaign data lands.
- R2 re-review needs PROOF-REDTEAM `status: passed`; round-1 bundle is committed (`b6747b3`) —
  gpd may restart round numbering (EXP-32); don't let it overwrite uncommitted state.

## 9. This handoff lives on `paper/paper-a-draft`

If the campaign session works from main:
`git checkout paper/paper-a-draft -- .planning/HANDOFF-CAMPAIGN-PHASE2-SUBMIT-20260703.md`
