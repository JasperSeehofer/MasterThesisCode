# Project State

A short, human-readable snapshot of where the project is, for continuity across sessions and
machines. Detailed, ephemeral working notes are kept out of the repository by design (see the
`.gitignore` note on internal planning state); this file is the curated, durable surface.

**Last updated:** 2026-08-12

> Refreshed after a 17-day gap (previous entry 2026-07-26). The 2026-07-26 milestone content is
> preserved below under "Milestone 2026-07-26" and "Resolved / prior scoping"; everything under
> "Current focus" and "Next" is new.

## What works today

- End-to-end EMRI simulation → SNR + Cramér–Rao bounds (GPU / cluster), and the CPU Bayesian H₀
  inference pipeline over the GLADE+ catalogue with completeness correction.
- Full CPU test suite green; `ruff` + `mypy` clean; docs and interactive figures deploy to GitHub Pages.
- `main` reflects the current, soundness-verified pipeline, including: zero-host pure-completion
  fallback, deep (z ≤ 1.5) population support, peculiar-velocity marginalization in the host-z
  kernel, an exact semi-analytic 2D denominator, and a value-preserving batched/fused likelihood
  evaluation (~3.8× faster).
- **A pre-registered, self-auditing research process**: `docs/RESEARCH_CYCLE.md` (through
  amendment A6), the physics-gate ledger `docs/gates/PHYSICS-GATE-LEDGER.md`, blind-locked band
  registrations, and adversarial adjudication. Three threads have now been closed *against* the
  programme's convenience using it (rows 96–98).

## Current focus

**Thread 17 — venue transfer, the running decider.** Does the σ_z-dosed coverage collapse seen
in the realistic ball venue transfer to a production-matched venue?

- **The calibration gate was falsified and rebuilt, not patched.** v1 self-voided
  (V4 texture band + DS-7 fired ⇒ GATE-NOT-TRUSTWORTHY, `3a572897`). v2 was re-registered on
  **disjoint seeds** — zero overlap with v1's `20260808+[0,9049]` envelope — with the V4 band
  re-derived from a pre-declared analysis, DS-7 demoted to report-only, and the clean rule
  enforced (`065e7f58`). v2 came back TRUSTWORTHY with no §10 trigger.
- **v2's verdict is KEEP-DIGGING via clause (b), DEFECT-class** (`64abd5f6`, ledger row 98,
  cluster array 6250988). Six gate-weighted decision cell×channels FAIL both DS-1 (0/0/0) and
  DS-2 (KS D ≈ 1.000); MAP bias +0.0349…+0.0374 against post_sd_median 0.0012–0.0059 — the
  posteriors are roughly an order of magnitude too narrow relative to their own bias. Three
  pre-registered pattern-reproduction targets CONFIRMED on disjoint seeds (DS-8 T1 starvation
  rail 400/400 at all three truths · T2 ball-venue +σ_z bias, all 8 banded components inside
  v1 ± 4√2·SE · T3 B0 exactly on truth 400/400 both channels).
- **Decider running.** `results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md`
  (`e77eecad`) registers the production-matched ball venue as the discriminating measurement.
  Campaign array **6259842** (final 22 chunks, 24 h limit) completes overnight 08-12/13.
  Branches: TRANSFER-CONFIRMED (⇒ `/physics-change` intake on estimator photo-z handling,
  author-gated) / TRANSFER-REFUTED / MIXED / VENUE-CONFOUNDED. Resume recipe: runbook 9 §6.
- **Two threads closed honestly since the last entry.** Cross-term (Eq. 31):
  NEGLECT-WITH-NUMBER in all four venue/channel cells, with a re-evaluation trigger register
  (row 96). M-2 residual owner: C1 confirmed (+0.021–0.022 nat/event, cluster-robust),
  instrument B MIXED, thread closed as **ratified dissolution** — confounding-absorbable,
  family undecidable by matching (row 97).
- **Track B — HPC performance deep-dive**, author-mandated 2026-08-12, on branch
  `perf/realistic-venue`. Hotspot is the h-independent φ(M) chain (76% of seed time). Landed:
  `--grain h` bit-identical event-level parallel mode (`082d1e07`, certified); φ(M) two-segment
  affine swap (`87c6670b`) and Route 1 adaptive Gauss–Hermite order (`dfedf19c`), both
  `[PHYSICS]`, both ratified 2026-08-12 (`7c58f31e`). Roadmap:
  `results/venue_transfer_20260811/perf/PERF_ROADMAP.md`.
- **Rebrand in flight → `darksiren-emri`.** GitHub repo renamed and Pages URLs verified live
  (`e455572e`); Python package renamed `master_thesis_code` → `darksiren_emri` (`227e7a32`).
  Deferred to a coordinated window: the local directory rename and the cluster repo rename
  (ONE-repo rule requires them together) — see `docs/REBRAND_MIGRATION_CHECKLIST.md` and the
  prepared `scripts/migrate_local_rename.sh` / `scripts/migrate_cluster_rename.sh`.

### Author ratification bundle (open, queued 2026-08-12)

1. Items (i)–(vi) of the 08-11 continuation record (ledger §5 "AUTHOR CONTINUATION"): deviation
   register D1–D8; DS-8 quotability; thread-17 co-candidate; venue-transfer as decider; DS-7
   open; #47 wording.
2. Venue-transfer §11 deviation notes ×3 (runtime blowout + resubmission; V-T5 sequencing;
   contention resubmission 6259842).
3. Standing queue: issue #53 (3σ window); N-2 adoption; ledger renumber (recommend §1b fork);
   `LITERATURE_WARNINGS` H-g status vocab; DS-5 F5 matched-population.

## Next

- **Track A: retrieve and adjudicate 6259842** → collect/integrity → band-scored readout against
  the pre-registration → independent adjudication → ship the verdict. This is the next real gate.
- **T-1, the blind sealed-h mock, is still unrun** — the discriminating experiment for
  `H-MTC-genmarg`, with its kill criterion already written, 17 days after the programme named it
  decisive. Roughly 25 commits of protective-belt work have landed on top of the closure it is
  meant to certify. The "closes at truth" headline is what is at risk here.
- **Thread-level kill criteria are now on the record** (approved 2026-08-11): anti-tuning (T-1
  closes off the sealed h_inj ⇒ the `generator_marginal` adoption reverts and the closure claim
  is withdrawn, not patched) · precision (photo-z kernel degrades σ_h to the host-known limit
  ×3.3–×6.8 or worse ⇒ all precision claims restated as mock-internal) · time-bound (no
  blind-mock verdict by **2026-09-23** ⇒ ship as a *methods* paper with the closure reported as
  unblinded).
- **Paper #47** is on hold: "P–P leg FAILED — coverage DEFECT" (pending confirmation item vi).
  Verdict-independent sections are writable now — methods (pre-registration discipline, gate
  design) and the dissolved-threads narrative (rows 96–97 are publishable regardless of the rail
  outcome).
- Workspace note: `ws_extend` used the **last** available extension (expires **2026-09-23**) —
  copy finals off before then. This is also the kill-criteria time-bound.

### Resolved since the last entry

- **Redteam adversarial review: merge gate CLEARED** (`results/redteam_20260726/CONSOLIDATED_VERDICT.md`).
  Math review SOUND-WITH-CAVEATS; **no evidence of tuning to h = 0.73 in any estimator constant**.
  The caveats rescope claims and order follow-ups rather than blocking.
- **seed900 fixpool re-run** — completed the registered n = 5 multi-seed test; the campaign
  verdict already stood on the valid-4 basis.
- **P–P impostor-capable harness extension** — merged (PR #49, `feat/pp-impostor-harness`); it
  is the harness the calibration gate now runs on.
- **Evaluate-path instrumentation** (`9522467`/`b287670`/`234890f`, 2026-07-31): per-class per-h
  Σ ln p_i logging on both channels, w_G at 7 s.f., P6 host-recovery counter with pruned-frame
  index translation. Output-invariant, verified. Caveat kept: the P6 scattered path assumes
  injection and evaluation share M_min/M_max/z_max — true for the standard `main.py` workflow,
  documented inline.

## Known open questions (tracked honestly)

- Whether the H₀ posterior peak is fully information-driven versus partly a normalization effect in
  the deepest-incompleteness regime — under investigation (`docs/H0_BIAS_RESOLUTION.md`).
- Whether the ball-venue coverage collapse is a venue artefact or a production-relevant defect —
  thread 17, decided by array 6259842.
- Two alternative host-kernel normalization modes (`volume_trunc`, `mass_trunc`) were implemented
  and empirically rejected; they are retained as documented, non-default experimental modes.
- Standing modelling choices are documented in `docs/source/limitations.rst` (flat-ΛCDM distance
  integrals, cosmology-constant vintage matched to the mock universe, redshift-uncertainty scaling).

## Milestone 2026-07-26: the deep venue closes at truth

- **`generator_marginal` + `--pdet_z_resolved`** (branch `physics/absolute-mass-marginal`,
  `[PHYSICS]` commits `8fbb21e` + `a608c4f`): seed1000 MAP = **0.7300 = truth in both
  channels**, sharp broad-based peak, 1017 tests green. Fully derivation-backed chain
  (generator-consistent normalization n̂_w = W_cat/V_f + D_gen; z-resolved survival
  S(d_L|z) in u = ln(1+z); point/point pairing verified against the generator — the mock
  draws catalogue z verbatim). Two-sided mechanism validation: FIX-2-alone measured
  −68.75 ln vs −69 predicted. No truth-referencing constants anywhere. For REAL data the
  photo-z kernel must return (point/point is generator-exact for the mock only).
- **Campaign NO-GO LIFTED (2026-07-26), on the valid-4 basis.** The five-seed production-stack
  campaign (jobs 6044799–6044808, code @ `6dae9d3`, 41-pt grid) passed all pre-registered
  criteria on seeds 1000/2000/3000/90000: bias −0.0003 ± 0.0004 (base channel), width
  χ² = 8.0/3.7 both VALID, MAPs interior, both channels clean. seed900 was dropped from the
  registered set (author-ratified) for a diagnosed input-provenance defect — a mis-populated
  204-injection pool instead of the canonical `injection_pool_depth15_50k` — not an estimator
  failure. See `results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md`.
- **Production defaults flipped** (`[PHYSICS]` `ce6338e`): `--normalization_mode` default is now
  `generator_marginal`, `--pdet_z_resolved` defaults to `True` (`--no-pdet_z_resolved` for legacy
  pooled behaviour). Library defaults match.

## Prior scoping (2026-07-25, kept for the record — resolved by the milestone above)

- **Deep-venue rail (issue #30, closed)** — the EXP-40 re-evaluation confirmed the seed1000
  posterior railed at the lower grid edge (MAP h = 0.60, both channels). Post-fix diagnostics
  re-attributed the rail: ~82% of the tilt is the host-found `L_cat` term, and z ≤ 0.3 subsets
  still railed — weakening pure depth truncation, strengthening the L_cat/Gray-mixture estimator
  path. See `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`.
- **z_cut truncation scan (2026-07-25): ALL RAIL.** Consistently truncated re-evals at
  z_cut ∈ {0.2, 0.3, 0.5} on the same seed1000 CRB all railed at h = 0.60 in both channels —
  depth truncation (issue #30 option b) is empirically dead. The untruncated z ≤ 0.2 *subset*
  closes at 0.729 while the truncated z_cut = 0.2 re-eval rails, isolating the rail in the
  h-dependence of the truncated selection/normalization structure (w_G = β_G/D) interacting with
  L_cat.
- **Rail mechanism (2026-07-25): host misassociation.** Two independent investigations
  (empirical per-event decomposition, validated to ≤ 4.5e-13 against cluster diagnostics;
  structural audit vs Gray A9 / Gair 2023 / gwcosmo v2) showed 91–100% of each rail event's tilt
  is the numerator GW-likelihood × host-z overlap: candidate balls contain only foreground
  galaxies (preferred h* ≈ 0.42–0.48, below the grid). volume_deconv is exonerated (exactly
  h-invariant); the ball-local selection denominator is a real-but-secondary discrepancy vs the
  references (1–14%). See `results/lcat_h_dependence_20260725/SYNTHESIS.md`.
