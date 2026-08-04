# Runbook — next session (written 2026-08-04)

Supersedes `RUNBOOK_NEXT_SESSION_6.md` (its fix program is DONE: both fixes derived,
adversarially verified, author-gated, implemented, landed on `main@658c428a`, and
evaluated post-fix). Session ran on the dev box; cluster is synced to the same commit.

**Read in this order:**
1. `.planning/derivation-2dbias-fix-20260803/FIXB_PATHA_PACKAGE.md` — the ratified
   path-A package, §8 author decisions D1–D3 (all decided 2026-08-04, see ledger).
2. `.planning/derivation-2dbias-fix-20260803/fixb_measurements/FIXB_MEASUREMENT_REPORT.md`
   — the r_φ verdict (path A mandatory) and the attribution campaign.
3. `$WS/run_20260804_postfix_iiib/RUN_NOTES.md` + `run_20260804_postfix_joint_r1/RUN_NOTES.md`
   (cluster) — the post-fix evaluates: config, era decision, pins, results.
4. `docs/gates/PHYSICS-GATE-LEDGER.md` — complete presented/implemented/verified trios.

---

## 0. State of the physics (2026-08-04)

Both `/physics-change` fixes are landed and **verified arithmetically correct — and
insufficient**:

| channel | pre-fix | post-fix MAP | injected | pre-registered pin |
|---|---|---|---|---|
| 1D idealized | 0.7450 | **0.600 RAILED** | 0.730 | 0.600 railed ✓ |
| 2D idealized | 0.7900 | 0.780 (peak-local mean 0.7803) | 0.730 | 0.7815 ± 0.0160 ✓ |
| 1D joint r1 | 0.7400 | **0.600 RAILED** | 0.730 | 0.610 off-rail ✗ (by a hair) |
| 2D joint r1 | 0.8133 | 0.800 (peak-local mean 0.7907) | 0.730 | 0.7909 ± 0.0119 ✓ |

- The estimator lands on its own pins to 5 s.f. (r_Malm 0.3827762 parent /
  0.4415122 observed; w̃_G 0.06196684/0.07080225; legacy w_G bit-identical 0.1215039).
- **Full-grid** posterior means are ≈0.723–0.728 ± 0.068 in all four channels — the
  posteriors are near-flat with local structure. Distinguish peak-local means from
  full-grid means in every readout; they differ by up to 0.07.
- The 1D rail-at-0.600 was the pre-registered ADVERSE branch: 1D's old ~0.74
  centredness is now *proven* to have been a contingent cancellation, not physics.
- **The residual 2D bias (+0.05–0.07 at MAP) survives at the idealized venue** with
  exact host redshifts and no scatter ⇒ owned by something the fixes didn't touch.
- NOT the STOP branch (2D moved toward truth, onto its own prediction).

**Prime remaining suspects, ranked:**
1. **D1 — the p0-window mass band-pass** [NEW, discovered 2026-08-04]: campaign-51
   detections were selected by SNR ≥ 20 ∧ p0 ∈ [10.002, 15.998] (stale snapshot-era
   `ParameterSpace.p0` bounds vs the ratified plunge-window ICs `e419062c`), removing
   69.3% of SNR-passers mass-dependently; no inference selection object models it.
   A mass band-pass is a z-selection distortion at fixed source mass — exactly the
   shape needed to own a 2D-channel residual. S_and instrument exists
   (s_G/s_D = 0.7305 ± 0.4%, `fixb_x15_attribution/`).
2. **The in-cat class tension** (class argmax ≈ 0.83 post-fix): the C5/C7
   completion-admixture track; path A explicitly did not close it.

## 1. Task queue (in order)

1. **Gate (vii) proper read** — the post-fix dark-class catalogue-leg channel
   difference `Σ_dark Δln(L_cat^2D/L_cat^1D)` (HEAD baseline −504.8 nats, 0.73→0.81)
   via the `cellb_readout.py` conventions over the NEW 41-h diagnostics CSVs (both
   post-fix runs emit `simulations/diagnostics/event_likelihoods.csv`, 41 h × 1588
   events, 7 s.f. path-A columns). Crude whole-mixture read was +6.45 (joint) /
   +8.15 (idealized) nats — NOT the same object; do not conflate. [haiku/low compute,
   opus/high interpret]
2. **D1 investigation** — is the p0-window a 2D-bias owner?
   (a) Counterfactual: re-score the 2D channel with S_and-consistent selection
   objects (instrument exists) — signed, sized effect on the 2D MAP.
   (b) The bounds retirement (`ParameterSpace.p0`) is its own SMALL `/physics-change`
   (simulation-side; 5-item gate, then re-simulation for future campaigns only —
   the existing 3135-event catalogue stays band-passed and must never be re-scored
   against band-blind objects).
   Check `BIAS_HISTORY_LEDGER.md` §2 before opening, per standing rule.
3. **C7-adjunct** (Σ_glob point → D_g smearing) — LAST, alone, per binding ship
   order; afterwards re-measure r_Malm(h) and re-score Fix B's gate (ii).
4. **D2 promotion** — truth-convention pins become primary only after truth Σ⁴ᴰ(h)
   is measured at all 41 h on the D1-remedied rerun (author decision of record).
5. **The calibration gate** (see §2) — before any further mechanism hunt beyond D1,
   run the two-channel SBC/coverage protocol; its verdict adjudicates
   keep-digging vs stop-and-report-bound.

## 2. META-TASK: establish the standing Research Cycle

**Author mandate (2026-08-04):** codify a structured research cycle for this project
so every future investigation follows one standardized step instead of reinventing
its runbook. The pieces exist and are battle-tested; the cycle chains them:

| stage | name | question | existing asset |
|---|---|---|---|
| 0 | **Claim intake** | what exactly is claimed, with what provenance? | claim files + tags ([LOCAL, VERIFIED]…), exoneration list, `BIAS_HISTORY_LEDGER.md` |
| 1 | **Information forecast** | what would perfect analysis of this data say? pre-register expected σ | F5 σ_z/σ_M engine; Fisher forecasts |
| 2 | **Pre-registration** | hypotheses, decisive reads, outcome branches, STOP signals, calibration bands — written BEFORE running | `PREREGISTRATION_2x2_cellB.md` pattern |
| 3 | **Measure / refute** | Gates A–C: provenance → adversarial refutation → alternative causes; model/effort policy; measurement-before-gate when a measurement decides a formula's shape | RUNBOOK-6 §1–§5 pattern; `/commission --research` |
| 4 | **Calibration gate** | SBC/PP coverage of the FULL two-channel estimator on truth-known synthetic universes at production venue + generator-closure absolute-count audit + forecast-consistent width | `validation/pp_coverage.py` (extend to 2-channel + realistic host-observation model); the (ii-d)-style count audit |
| 5 | **Decision** | CALIBRATED+narrow → measure; CALIBRATED+wide(≈forecast) → **stop digging, report bound**; DEFECT (≥3σ coherent class displacement or coverage failure) → fix via `/physics-change`; UNDETERMINED → the one measurement that decides | `/physics-change` + gate ledger; author gates |
| 6 | **Chronicle** | ledger rows, claim-file writebacks, next runbook | this file's lineage |

Stop/continue rule of record (author-endorsed 2026-08-04): the per-event
ln-posterior min/max range is a *screen*, never a stopping gate — N coherent
sub-threshold tilts dominate the ensemble (measured: per-event 0.3–0.5σ rails vs
+3.4–6.1σ class-summed). "Stop digging" requires: coverage pass + width on the F5
forecast + no unmodeled selection between generator and estimator (the D1 class of
defect — SBC alone cannot catch a filter both sides silently share; the
absolute-count audit is the complement that caught it).

Deliverable: `.claude/skills/research-cycle/SKILL.md` (+ a short
`docs/RESEARCH_CYCLE.md`) wiring the stages to the existing assets, so `/research-cycle`
is the entry point for every new investigation. Establish it BEFORE starting task-queue
item 2's mechanism hunt, then run D1 through it as the first full cycle.

## 3. Gotchas (new this session — several supersede RUNBOOK-6 §9)

- **`posteriors_fixed/` does not exist on the cluster** — RUNBOOK-6 §9's
  "posterior-directory trap" is a LOCAL-tree fact only. Cluster canonical dirs are
  plain `posteriors/` + `posteriors_with_bh_mass/` per sub-run-dir; the stale era is
  quarantined by filename (`*_PRE_ec09ed0.bak`). The only regular CRB files in the
  seed-61000 tree are the two under `simulations/`; everything else is symlinks.
- **rsync of cluster run-dirs copies symlinks as symlinks** — the injection-pool
  "copy" in the run dir is links into `$WS` root. Always `rsync -L` from the real
  source and verify a content fingerprint (the pool: 707 files, 200100 data rows,
  `dist(1.3261748578964083, 0.73) = 9.164987 Gpc`).
- **Combined posteriors land inside each channel dir**, not at `simulations/` root.
- Per-h evaluate cost at 1590 events: ~4–5 min (vs 56–76 min at 3355) — full 41-h
  campaigns finish within the hour; don't over-provision walltime.
- Data staged locally (workspace-expiry risk RETIRED for Fix-B needs):
  `results/campaign51_20260728/realistic_20260729/realizations_staged/`
  (observed_catalogue_seed900001.csv + meta + cluster-parent catalogue) and
  `.../gate_b_20260730/injection_pool_mix200k_20260728/`. Workspace expires
  **2026-09-23**; anything else needed off it must be copied before then.
- The w_G pin "0.86 = 0.1038732" in GATE_PACKAGE_FINAL §1.5 is an h=0.81 mislabel
  (run log: w_G(0.86) = 0.0947). Fix A's tests use the correct labels.
- `gen_ch03.py` and `step3_instrument.py` call the old `child_process_init`
  signature → they reproduce the PRE-C7 kernel (implicit but arguably correct:
  they reproduce published numbers). Make explicit when next touched.

## 4. Author decisions open

- RUNBOOK-6 §10's "fix HA knowing it worsens the MAP" now has its direct evidence
  (1D railed at 0.600 post-fix). The paper-facing question has sharpened to:
  **which channel, which convention, and is the headline a measurement or a bound?**
  — route through the §2 calibration gate rather than deciding on instinct.
- RATIFY-R7 (extra GPU truth seeds): unchanged; feeds the calibration gate's
  seed count (truth-seed scatter 0.023 vs realization scatter 0.006).
- Paper (#47) remains ON HOLD pending a trusted run — the calibration gate is now
  the explicit gatekeeper for "trusted".
