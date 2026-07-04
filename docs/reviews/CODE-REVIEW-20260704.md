# Deep code review — 2026-07-04

Weekend Workflow-orchestrated review of the whole codebase, run while the Phase-2 GPU
campaign self-drove on bwUniCluster (2FA-blocked until Monday). Scope: every `.py` under
`master_thesis_code/`, the test suite, and `cluster/*.sh|*.sbatch`. Branch
`review/codebase-20260704` (stacked on `physics/campaign-depth-pv`).

Method: 12 dimension reviewers (schema'd findings: id, file:line, severity,
campaign_safety, quoted evidence, proposed fix) → dedup → adversarial verification of
majors → coverage critic. The first workflow pass exhausted API credits partway
(dimensions 1–3 and the verifier pass died); a second pass re-ran the three dense physics
dimensions (likelihood-core, selection-function, catalogue-handler) plus the coverage
critic. 77 raw findings, 72 after (file,line) dedup.

## TL;DR

**The physics core is sound.** The three densest dimensions found **no critical or major
defects**. Gray et al. (2020) A.9/A.10/A.19 faithfulness confirmed; the suspected
"quadrature dilution" over the deepened volume is **not** borne out (Gauss–Legendre n=100
resolves the completeness feature to ~1e-14 rel. error even at z_max=3); the MC importance
sampler still has proposal = prior after the G2d Eddington-M shift (mass cancels exactly);
worker-global lifecycle is clean (forkserver, fresh pool per `evaluate()`).

Everything landed this weekend is **campaign-neutral** (docs, guards, dead-code removal,
metadata, test additions, cluster-script trivia). Everything that would move a posterior
number or change what the GPU campaign computes is **deferred to a GitHub issue** — nothing
semantic was touched while evaluations run.

### Commits on this branch

| commit | what |
|---|---|
| `393666b` | docs: CLAUDE.md + hpc-gpu.md Known-Bugs / Pipeline-A reconciliation (DED-01/02, HPC-01, PHY-04/06) |
| `562918e` | ops(cluster): trivial robustness — CLU-03/06/07 (Monday deploy) |
| `26ce5d4` | docs(pv-test): §7b PV-value-correction bound + reproducibility (issue #16) |
| `7a41603` | fix: simulation-loop robustness + run-metadata provenance (SIM-01/02/03/04/07, REP-02/03) |
| `3e7f034` | test: end-to-end GLADE+ catalogue writer/reader contract (TQ-01) |
| `8c789a6` | fix: wCDM guard on the analytic distance (PHY-01/02, #4) |
| `90bd40e` | refactor: remove Pipeline-A dead code (DED-04/05/06/07) |
| `a712a02` | fix: minor safety + stale-comment cleanup (PHY-07/09, HPC-06) |
| `a1e151f` | fix(plotting): truth-h from constants, un-clip h-axis, stale-comment sweep (PLT-03/04/05/10, TQ-04) |

**Not landed (minor, deferred):** the GPU-perf nits HPC-02 (default `use_gpu=True` in
constructors), HPC-03 (unbounded PSD cache), HPC-04/05 (needless GPU round-trips for host
scalars) are real but exercise the GPU path (hard to verify on the CPU dev box) and are
low-value; HPC-07 (dead T-channel `S_zz` parenthesis) is a physics formula and needs
`/physics-change`. LC-01 (stale MC-weight comment) folded into issue #24. TQ-02/03/05/07/08
(test-coverage improvements) and PLT-02/06/07/09 (figure provenance, dead callbacks, Plotly
template) are backlog polish, not landed tonight.

## Campaign risk assessment (read first)

The recon flagged **one** item that could affect the *running* campaign: eval-node OOM at
depth 1.5. **It is refuted.**

- **Catalogue RAM is ~flat in depth.** Direct count on the 22.64M-row reduced CSV:
  mass-bearing rows at z≤0.5 = 21,742,890 vs z≤1.5 = 21,753,293 — the deepening added
  **~10.4k rows (+0.05%)**, not the assumed 4–5×. GLADE+ is a *local* catalogue; essentially
  all mass-bearing galaxies already sit at z≤0.5, so the z-prune keeps ~21.75M rows either way.
- **The catalogue is not forked across workers.** The multiprocessing pool uses
  `forkserver` (fallback `spawn`), and `galaxy_catalog` is **not** in `child_process_init`'s
  `initargs` (only small per-detection arrays + `D_h_table` + the p_det object are). The
  ~4–5 GB DataFrame + two BallTrees live **once** in the parent; workers hold ~0.3–0.6 GB
  of preloaded modules + pickled detection arrays. Memory is ~flat in event count, so the
  campaign's 2–4× event yield does not scale the dominant term.
- The 6h/16-CPU eval already completed at these settings (job 5732036).

**No Monday OOM fix is required.** The only related residual is that **no sbatch sets
`--mem`** (memory is the implicit partition default everywhere) — a belt-and-suspenders
gap tracked as CLU-05 / issue #27, to be pinned from `sacct MaxRSS` on Monday, not an
active hazard.

One secondary campaign-ops caveat surfaced: **the unattended drift monitor is blind**
(CLU-02) — `combine.sbatch` saves the baseline before comparing to the same path, so the
drift check compares each run against itself. Treat rsynced `comparison_*.md` as
uninformative until issue #27 lands on Monday.

## Findings by disposition

Severity: critical / major / minor / nit. Campaign-safety: **sim** (changes GPU-computed
quantities), **inf** (changes posterior numbers), **neutral** (refactors/guards/docs/tests).

### Landed this weekend (neutral only)

| id | sev | file | what | disposition |
|---|---|---|---|---|
| SIM-01 | major | main.py | 90s SIGALRM never cancelled on data_simulation exception paths → stale alarm can kill an unattended task | try/finally alarm-cancel |
| SIM-02 | major | main.py | `warnings.filterwarnings("error")` leaks across iterations; filter list grows unbounded | scoped `catch_warnings` |
| SIM-03 | major | main.py | 9 swallowed exception classes with zero per-class counting → unquantified selection effect | per-class skip counters in end-of-run summary |
| SIM-04 | major | main.py | injection_campaign has no SIGTERM flush → up to 1999 SNR evals lost on wall-cap | SIGTERM flush handler |
| SIM-07 | nit | main.py | `raise ValueError(e)` re-wraps and mangles the original exception | bare `raise` |
| PHY-01 | major | physical_relations.py | `dist()` accepts w0/wa but always evaluates ΛCDM (silent wrong result for wCDM) | NotImplementedError guard (neutral; every production call uses defaults) |
| PHY-02 | minor | physical_relations.py | `dist_derivative` ignores its cosmology args | forward args |
| PHY-07 | nit | cosmological_model.py | `DarkEnergyScenario.de_equation` divides by w_a (CPL should multiply); ZeroDivisionError at fiducial | fix formula / delete (dead) |
| PHY-09 | minor | parameter_space.py | M `derivative_epsilon` comment miscalculates the log-uniform midpoint 100× | correct the comment (no value change) |
| REP-02 | major | main.py | run_metadata `cli_args` omits the inference-critical flags (normalization_mode, pdet_*, …) | serialize the full parsed namespace |
| REP-03 | minor | arguments.py | `seed` property returns a fresh random value per access; combine records a phantom seed | cache the draw |
| DED-01 | major | CLAUDE.md | documents Pipeline A files deleted in c1571a2 | reconciled |
| DED-02 | major | CLAUDE.md | Known Bug #1 (unconditional cupy import) is false — import is guarded | struck |
| DED-04..07 | minor | (multiple) | dead code: `datamodels/galaxy.py`, `parse_to_reduced_catalog_with_reduced_errors`, `single_host_likelihood_grid` stub + legacy global, dead constants, `scripts/quick_snr_calibration.py` | one coherent deletion |
| HPC-01 | minor | CLAUDE.md, hpc-gpu.md | stale "unconditional cupy import" claim (fixed 4894648) | reconciled |
| HPC-06 | minor | LISA_configuration.py | unknown-channel PSD fallback returns silent all-zero PSD + hardcodes np | raise ValueError |
| PHY-04/06, TQ-04, PLT-03/04/05/10, DED-09 | nit/minor | (docs/plots) | stale docstrings, hardcoded truth-h literals, tension-explorer x-range clips h=0.86, Mpc/Gpc label, stale line-refs | doc/plot fixes |
| TQ-01 | major | (test add) | GLADE+ catalogue writer/reader has no end-to-end test | new `test_handler_catalog_io.py` |
| CLU-03/06/07 | nit/minor | cluster/* | `\|\| true` on grep-miss branches under set -e; pkill bracket idiom; watcher run-dir glob widened | landed |

### Deferred to GitHub issues (semantic — not touched while the campaign runs)

| id(s) | sev | issue | why deferred |
|---|---|---|---|
| COM-01, COM-03, COM-04 | major/minor | **#23** | completion-term realism at depth 1.5 (double-count; luminosity- vs rate-weighted f; K-corr). Self-consistent within the injection closure → campaign-neutral; Paper-B design notes. |
| SEL-01, LC-02, SEL-02, CAT-02, DED-08 | minor/nit | **#24** | p_det / with-BH-mass corrections (M_z kernel exponent N^-1/6 vs N^-1/5; MC z-clamp; NaN guard; candidate window narrower than kernel; d_L clip under depth-1.5). Inference-semantic → post-harvest `/physics-change` batch. |
| PHY-03 | major | **#25** | `get_redshift_outer_bounds` ignores `sigma_multiplier`/Omega_m bounds (2σ requested, 3σ delivered). Neutral first step (preserve numbers) can land; 2σ decision is inference-semantic. |
| REP-01 | major | **#26** | emcee proposal RNG unseeded → injection (M,z) not reproducible under `--seed`. Sim-semantic; changes which events the campaign draws. |
| CLU-01, CLU-02, CLU-05, CLU-09 | major/minor | **#27** | cluster orchestration robustness (double-submit on cleanup; blind drift monitor; no `--mem`; combine afterok zombie). Land scripts in the PR, **deploy Monday** — do not restart live jobs. |

Reconciled existing issues: **#5** (Pipeline-A 10% error) and **#6** (WMAP cosmology) closed
as moot/design-choice; **#4** (wCDM), **#7** (galaxy.py (1+z)³), **#8** (two-pipelines)
commented with the Pipeline-A-removed reconciliation.

### Verified-correct (no finding — recorded so it isn't re-litigated)

- **likelihood-core**: Gray A.9/A.10 numerator (GW×prior, no p_det) / denominator (p_det×prior)
  partition faithful; ratio-of-sums `L_cat=(Σ w_g N_g)/(Σ w_g D_g)`, `p_i=(β_G L_cat+B_num)/D(h)`;
  fixed_quad n=100 machine-precision over z_max∈{0.5,1.5,3.0}; MC proposal=prior post-G2d
  (mass cancels); forkserver worker lifecycle leak-free.
- **selection-function**: `searchsorted` side conventions consistent; **h-invariance holds
  end-to-end** (d_hor from stored injection d_L, no h leak); shallow-pool/z_cut/empty-pool
  gates thread correctly; M_z clamp is inside the padded axis (safe).
- **catalogue-handler**: COORD-03 ecliptic rotation symmetric for build+query; HostGalaxy
  field provenance correct; OOM premise refuted (above).
- **physics/cosmology**: `dist_to_redshift` fsolve robust and round-trip-exact to 25 Gpc
  (depth-1.5 safe); 5-point stencil coefficients correct; rate density strictly positive
  over the sampled domain; comoving-volume-element dimensionally consistent.
- **hpc-gpu**: all 4 GPU modules already guard cupy; xp pattern respected; `scalar_product`
  does exactly one device→host transfer per call; memory freed per step not per inner loop.

## Coverage

Across both workflow passes the reviewers read essentially the entire production path:
`bayesian_statistics.py`, `simulation_detection_probability.py`, `handler.py`,
`pixel_completeness.py`, `main.py`, `physical_relations.py`, `cosmological_model.py`,
`parameter_space.py`, `galaxy.py`, `constants.py`, `parameter_estimation.py`,
`LISA_configuration.py`, `memory_management.py`, `decorators.py`, `waveform_generator.py`,
`arguments.py`, `posterior_combination.py`, `detection.py`, `pp_coverage.py`, all
`plotting/*`, `callbacks.py`, `dark_siren_injection.py`, and the active `cluster/*`.

**Genuine coverage gaps** (not deeply read; both sim-semantic so any finding would be an
issue not a fix while the campaign runs): `emri_rate.py` (M1 intrinsic rate density —
physics-critical, shared by sim + prior) and `__main__.py` (CLI teardown / `os._exit`
figure-hang workaround). Recommend a targeted read of `emri_rate.py` in the next
physics-change window. Analysis-side `parameter_estimation/evaluation.py` and the
`scripts/bias_investigation/*` one-offs were intentionally skipped (off the live path).

## §7b — isolated PV value-correction bound

See `results/pv_correction_test_20260703/ANALYSIS.md` for the full analysis (posted to
issue #16, which is now closed). Summary: on the frozen seed600 event set (the designed
worst case — all-low-z, where all 709k GLADE+ PV corrections live), removing the
peculiar-velocity **value** correction shifts the **1D** H0 estimate by Δmean(live−noPV) =
**−0.0142** (−3.3 posterior widths), ΔMAP = −0.010; the production-relevant **2D** (with BH
mass) channel is **PV-insensitive** (Δmean **+0.0012**, +0.2σ — consistent with zero, on the
un-railed 17-value grid; the BH-mass information dominates and is PV-independent). The
campaign-side σ_v **marginalization** (commit `8568d9f`, σ_v=200 km/s) covers this as added
uncertainty, and real campaign events at depth 1.5 are far less PV-sensitive than the
all-low-z seed600 worst case.
