# Runbook — next session (written 2026-07-29, session end)

Supersedes `../lcat_h_dependence_20260725/RUNBOOK_NEXT_SESSION_4.md` (its
thread 1, the #51 campaign redesign, was executed end-to-end this session;
its thread 2 (P1 parity audit) is DONE and RETIRED the −6.5 gate; threads
3–6 remain, re-listed at the bottom).

## 1. What is RUNNING right now (check these first)

All CPU-only, submitted 2026-07-29 ~18:40 on bwUniCluster, no GPU dependency:

| What | Jobs | Output path (on `$WS`) | Purpose |
|---|---|---|---|
| σ→0 end-to-end control | 6092510/11 | `run_20260729_seed61000/sig0_control/` | **Gate R6 pipeline half**: must reproduce the campaign-#51 seed-61000 posterior EXACTLY (baseline modes, byte-identical catalogue). If it does not, the realization plumbing is not inert — investigate before trusting anything below. |
| Realistic grid, 2 seeds × 5 realizations | 6092512–6092531 | `run_20260729_seed{61000,62000}/real_r{1..5}/` | Campaign #53 production (`absolute_marginal` × `volume_deconv`, observed catalogues `realizations_20260729/observed_catalogue_seed90000{1..5}.csv`) |
| 0.67 closure seed | (orchestrator-submitted) | `run_20260729_seed64000_h0p67/` | The one closure test the author asked for; feeds the idealized baseline readout's empty row |

`squeue -u $USER`; monitors from this session are dead (session ended).
**SSH gotcha:** the ControlMaster dies every few hours and then *hangs*
instead of failing; clear with `ssh -O exit bwunicluster; rm ~/.ssh/cm-*` and
re-auth with 2FA (`ssh bwunicluster echo ok`).

## 2. First actions next session

1. **Score the σ→0 control.** Compare
   `sig0_control/simulations/posteriors/combined_posterior.json` against
   `run_seed61000/posteriors_fixed/combined_posterior.json` (local mirror).
   Expect bit-identity. This is a ratified gate, not a nicety.
2. **Collect the 10 realistic posteriors** and produce
   `REALISTIC_READOUT.md`: per (seed, realization) MAP/σ/bias, the SPREAD
   across realizations, pull statistics vs truth (should be ~N(0,1) —
   MAP-on-truth is explicitly NOT expected any more), and the in-catalogue vs
   dark split. Pre-registered predictions P1–P6 are in
   `docs/derivations/realistic_host_observation_model.md` §8; score them
   honestly, including falsification conditions.
3. **The author's open decision** (deferred by design, [RATIFY-R7]): if the
   realization spread is dominated by *which hosts happen to be
   spectroscopic* (expected ~3.4 spec hosts/seed, Poisson), extra GPU truth
   seeds are needed for a stable headline. Decide with the measured variance
   in hand — do not pre-commit.
4. **Finish `IDEALIZED_BASELINE_READOUT.md`** (fill the 0.67 closure row +
   its zoom; the zoom recipe is `H_VALUES_OVERRIDE`, see §4 below).

## 3. State at handoff

- **Idealized baseline (RETAINED as consistency evidence, not a forecast):**
  seeds 61000/62000 recover h_true = 0.73 with σ_h = 3.0/3.9e-4 and biases
  −0.24σ/−0.36σ. Readout: `IDEALIZED_BASELINE_READOUT.md`.
- **Why it is not a forecast:** `idealization_audit/IDEALIZATION_LEDGER.md` —
  100 % of the information comes from 76 in-catalogue hosts, ALL photometric
  (median σ_z/z = 49 %), whose z is injected as truth and point-evaluated.
  Realistic forecast: σ_H0 ≈ 1.3–1.7 km/s/Mpc (~50× wider).
- **Physics fixed this session** (all `[PHYSICS]`, all with before/after pins):
  `ecb56d6` single-source mass boundary + clamp removal · `49251f3`
  confusion-noise TDI transfer (the big one: pre-fix SNR suppressed up to
  ~1100× above log10 M_z ≈ 6.2 — **issue #52 is the contamination ledger for
  every pre-fix dataset**) · `e419062` plunge-window ICs + T = 4.5 yr
  (official, replacing an inconsistent 5.0/4.0 pair) · `ec09ed0` measured-mass
  domain · `6eb86ad` wbh grid m-nodes 31→69 · `7b30d1f` realistic
  host-observation model.
- **Selection function:** `injection_pool_mix200k_20260728` (200,100 rows) —
  acceptance PASSED (median ESS 9088, W-frac ESS<500 = 0.077 %, reachable
  w̄ = 0.99841). Reusable by #53 unchanged (p_det is an object over TRUE
  quantities — the [RATIFY-R5] argument, and the reason #53 is CPU-only).
- **Full band [1e4, 1e7] source-frame is FINAL**, verified not assumed: the
  pilot's pre-registered narrowing rule scored NOT VERIFIED with detections
  out to detector m = 6.96 (`PILOT3_READOUT.md`).

## 4. Recipes worth not rediscovering

- **Zoom h-grid** (the corrected-physics idealized posterior is ~15× narrower
  than the production grid step):
  `--export=ALL,...,H_VALUES_OVERRIDE="0.72800 0.72810 ..."` on
  `evaluate.sbatch` (`0818ced`); 41 points at 1e-4 spacing resolved it.
- **Realistic evaluate:** `OBSERVED_CATALOGUE=<path>` +
  `NORMALIZATION_MODE=absolute_marginal HOST_Z_KERNEL=volume_deconv`
  (`7fd60bb`). The guards REFUSE point-kernel/`generator_marginal` on a
  scattered catalogue — that is intended, not a bug.
- **Make a realization:** `python -m master_thesis_code <dir>
  --realize_observed_catalogue --realization_seed S [--realization_sigma_scale 0]`.
  It is a 1.7 GB CSV round-trip: **submit it as a SLURM CPU job**, it will die
  if run in an SSH foreground (learned the hard way).
- **GPU queue:** `gpu_h100_short` stalled ~20 h on fairshare after a big
  injection burn; `simulate.sbatch` now lists all four GPU partitions
  (`a9abd9d`) and A100s started within 30 s. Widen a pending array in place
  with `scontrol update job <id> partition=...` — no resubmission needed.
- **`_INJECTION_COLUMNS` footgun:** `_flush_injection_results` writes with an
  explicit `columns=` list, so a row key missing from that list is **silently
  dropped** (cost us the pilot-3 provenance columns, `acaa0af`).

## 5. Remaining threads (from runbook 4, still open)

3. **(d2) derivation** — selection-side M scatter/truncation; the remaining
   owner of the ≈ +23 ln 2D residual together with (g1)-as-support-limitation.
   NB the −6.5 (d1) gate is RETIRED (`P1_PARITY_AUDIT.md`); the replacement
   pre-registered prediction is ≈ −1 to −2.5 ln.
4. **B_num residual-bias model** (runbook-3 thread 2, unchanged).
5. **#39 blind alternative-truth mock** — arguably wait for the #53 universe.
6. **#23 completion-term realism** [paper-blocker when the paper resumes].
7. **Paper (#47) still ON HOLD.** When it resumes: every pre-49251f3 number is
   suspect below ~3 mHz (issue #52), and the headline H0 must come from the
   #53 realistic run, never from the idealized baseline.

## 6. Housekeeping

- Workspace expires **2026-09-23** (last extension used) — copy finals off.
- Bulk mirrors are gitignored (`pool_mix200k/`, `*/injections/`,
  `run_seed*/prepared_*.csv`); canonical copies on `$WS`.
- Open issues: **#51** (campaign redesign, effectively delivered — close after
  the #53 readout), **#52** (PSD contamination ledger), #40/#23/#39/#47.
