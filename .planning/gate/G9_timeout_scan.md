# G9 Groundwork — Waveform-Timeout Prevalence Scan (Existing Logs Only)

**Date:** 2026-07-02 · **Branch:** `physics/derail-completion-4pi` · **Author:** commission subagent (read-only scan; no simulations run, no cluster state modified)

## 1. Purpose and method

The EMRI simulation and injection pipelines wrap every waveform/SNR and Fisher-matrix computation in a `SIGALRM`-based watchdog. An event whose waveform computation exceeds the alarm budget raises `TimeoutError` and is silently *dropped* from the sample. This constitutes an additional, unmodelled selection effect on top of the SNR threshold: writing the realised detection indicator as

$$
D_i \;=\; \mathbb{1}\!\left[\rho_i \ge \rho_\mathrm{thr}\right]\cdot \mathbb{1}\!\left[\tau_i \le \tau_\mathrm{max}\right],
$$

where $\rho_i$ is the event SNR, $\tau_i$ the wall-clock waveform-generation time, and $\tau_\mathrm{max}$ the alarm budget, the second factor is *not* represented in the detection-probability model $P_\mathrm{det}(d_L, M_z)$ used by the Gray et al. (2020, arXiv:1908.06050, Eqs. A.6–A.10) likelihood — unless the injection campaign that estimates $P_\mathrm{det}$ truncates the *same* events at the *same* budget (it does not: the budgets differ, see §2). This note quantifies the prevalence of these timeouts from existing logs, per stage and per run, and specifies the instrumentation needed to bin timeouts by EMRI parameters for the upcoming campaign.

Method: (i) extract the exact WARN strings from `master_thesis_code/main.py`; (ii) grep all local logs (`simulations/` tree, 262 files) and the local backup `~/data-backups/seed600_local_derail_20260702/`; (iii) grep the bwUniCluster workspace `$(ws_find emri) = /pfs/work9/workspace/scratch/st_ac147838-emri` over read-only SSH in the run directories named in the task, plus the injection directories that feed $P_\mathrm{det}$ and (as a supplementary baseline) the canonical seed400 run.

## 2. Exact log signatures (code references, current HEAD `67d039f`)

All timeout handling lives in `master_thesis_code/main.py`.

**Simulation pipeline (`data_simulation`)** — alarm handler at `main.py:349-350` raises `TimeoutError("Computation exceeded 90s timeout")`; the handler is installed at `main.py:352`.

- **SNR stage** (`signal.alarm(90)` at `main.py:475`, covering both the quick pre-screen SNR and the full SNR): catch site `main.py:536-538` emits
  `"Waveform/SNR computation timed out (>90s). Skipping event..."` (main.py:537)
- **CRB stage** (`signal.alarm(90)` at `main.py:554`): catch site `main.py:568-570` emits
  `"Cramér-Rao bound computation timed out (>90s). Skipping event..."` (main.py:569)

**Injection pipeline (`injection_campaign`)** — its own alarm handler at `main.py:644-645`; budget `_TIMEOUT_S = 30` at `main.py:681`, armed at `main.py:732`. Catch site `main.py:776-778` emits the *same string as the simulation SNR stage*:
  `"Waveform/SNR computation timed out (>90s). Skipping event..."` (main.py:777)

Two implementation-vs-message deviations found while extracting the signatures:

- **D1 (mislabel):** the injection-stage message claims `>90s` but the actual budget is **30 s** (`_TIMEOUT_S = 30`, main.py:681). The message text at main.py:777 (and the handler string at main.py:645, "Computation exceeded 90s timeout") is wrong for this code path.
- **D2 (budget asymmetry, physics-relevant):** the simulation grants 90 s per waveform while the injection campaign that *estimates* $P_\mathrm{det}$ grants only 30 s. An event with $30\,\mathrm{s} < \tau < 90\,\mathrm{s}$ survives in the simulation but is dropped from the injection set, so the timeout selection does **not** cancel between numerator (events) and denominator ($P_\mathrm{det}$) of the dark-siren likelihood, precisely for the slowest-waveform (long, harmonic-rich) systems.
- The SNR-stage and injection-stage strings are identical; within a log file they are distinguishable only via the `%(funcName)s` field of the log format (`data_simulation()` vs `injection_campaign()`), which the file logs do carry.

Older campaigns used the same phrase family (git archaeology: `ee402e1` introduced a 60 s CRB timeout, `d6b4ff2` extended it to the SNR stage; the substring `"computation timed out"` matches all historical variants and was used for the greps).

## 3. Sources scanned

| Source | Log files | Sim-loop logs | Any `timed out` lines |
|---|---:|---:|---:|
| Local `simulations/` tree (recursive, incl. `_archive_v2_1_baseline`, `cluster_run_*`) | 262 | 0 | **0** |
| Local backup `~/data-backups/seed600_local_derail_20260702/` (drivers_master.log, crux logs, sweep, drv_* run.logs) | 23 | 0 | **0** |
| Cluster `run_20260619_seed{500,600,700,800}_phase50` | 0 | 0 | n/a (no logs exist — cancelled campaign; dirs hold only `run_metadata_*.json`, empty `logs/`, and output CSVs incl. `simulations/injections`) |
| Cluster `run_20260620_seed500_phase50` | 276 (133 `master_thesis_code_*.log` + `logs/simulate_5094525_*.{out,err}` + eval/combine) | 49 | see §4 |
| Cluster `run_20260628_seed600` | 403 (118 `.log` + `logs/simulate_5273669_*`, `evaluate_5278753_*`, combine/merge) | 80 | see §4 |
| Cluster `injection_20260628-184208_seed700`, `injection_20260620-{211950,213449}_seed43000` | 80/80/560 task logs | all | see §4 |
| Cluster `run_20260516_seed400_phase50` (supplementary; canonical seed400 data) | 46 sim-loop logs | 46 | see §4 |

Key negative result: **every local log is an evaluation/posterior-combination/driver log; the waveform loop (and hence every timeout) exists only in the cluster logs.** Double-counting hazard verified and removed: the root logger tees WARNs to stderr, so `logs/simulate_*.err` duplicates the `.log` lines exactly (seed500: 1267 = 1267; seed600: 962 = 962). All counts below are from the authoritative top-level `master_thesis_code_*.log` files only.

## 4. Per-run counts (deliverable a)

"Iterations attempted" is the loop counter $Y$ from the last `"X / Y evaluations successful"` line per task log, summed over tasks (each loop pass logs once at `main.py:419`; it includes $d_L$-pre-screen skips, which log only at DEBUG, so the waveform-attempt count is $\lesssim Y$ — the quoted timeout rates are therefore mild *under*-estimates of the per-waveform-attempt rate). "Detections saved" is the summed final success counter $X$ (log-derived; task resubmissions can inflate it relative to the deduplicated CSV counts, e.g. seed500 memory records 1385 unique events vs 1551 log-summed).

### Simulation runs (SNR stage: 90 s budget; CRB stage: 90 s budget)

| Run | Tasks | Iterations attempted | Detections saved | SNR-stage timeouts | SNR timeout rate | CRB attempts (SNR ≥ 20) | CRB-stage timeouts | CRB timeout rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260620_seed500_phase50` | 49 | 122 146 | 1 551 | **1 267** | $1.04\% \pm 0.03\%$ | 1 646 | **1** | $0.061\%$ |
| `run_20260628_seed600` | 80 | 163 496 | 2 685 | **962** | $0.59\% \pm 0.02\%$ | 3 499 | **1** | $0.029\%$ |
| `run_20260516_seed400_phase50` (suppl.) | 46 | 108 439 | 1 252 | **1 267** | $1.17\% \pm 0.03\%$ | — (not tallied) | **2** | — |
| `run_20260619_seed*` (4 dirs) | — | — | — | no logs retained | — | — | — | — |

Context per-stage outcome tallies (seed500 / seed600, `.log` only): quick-SNR pre-screen fails 63 150 / 95 282; full-SNR threshold fails 55 034 / 5 590; SNR passes 1 646 / 3 499; ZeroDivisionError skips 3 580 / 2 591; FEW mass-ratio boundary log-lines 51 135 / 28 990. The halving of the SNR-timeout rate from seed500 (1.04 %) to seed600 (0.59 %) coincides with the $d_L \le 2$ Gpc pre-screen + quick-SNR pre-screen ordering in the seed600-era code, which diverts far/faint events before the expensive full waveform.

### Injection campaigns (30 s budget; these estimate $P_\mathrm{det}$)

"Attempted (post $z$-cut)" $=$ events stored $+$ non-timeout WARN skips $+$ timeouts; the $z > 0.5$ importance cut never reaches a waveform.

| Injection dir | Tasks | Events stored | Other WARN skips | Timeouts | Timeout rate |
|---|---:|---:|---:|---:|---:|
| `injection_20260628-184208_seed700` (h = 0.73) | 80 | 72 000 | 2 536 | **945** | $945/75\,481 = 1.25\% \pm 0.04\%$ |
| `injection_20260620-211950_seed43000` (h = 0.9) | 80 | 72 000 | 2 484 | **937** | $937/75\,421 = 1.24\% \pm 0.04\%$ |
| `injection_20260620-213449_seed43000` (multi-h sweep) | 560 | 504 000 | 17 019 | **6 611** | $6\,611/527\,630 = 1.25\% \pm 0.02\%$ |

The injection timeout rate is strikingly stable at $1.25\%$ across h-values and dates — consistent with a fixed population-model tail of slow waveforms being clipped by the 30 s budget.

## 5. Do the log lines carry parameter information? (deliverable b)

**No.** The timeout WARN is a fixed string with no interpolated values. Verbatim examples from `run_20260628_seed600/master_thesis_code_20260628_131025_h_0_73.log`:

```
2026-06-28 13:12:00,992 [main.py:531 - data_simulation()] Caught ZeroDivisionError during trajectory integration. Continue with new parameters...
2026-06-28 13:12:00,992 [main.py:419 - data_simulation()] 4 / 112 evaluations successful. (3.1989072844751574/min)
2026-06-28 13:13:30,993 [main.py:536 - data_simulation()] Waveform/SNR computation timed out (>90s). Skipping event...
```

```
2026-06-28 13:14:24,772 [baseclasses.py:746 - sanity_check_init()] Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q=0.00011213098401057018)
2026-06-28 13:15:54,773 [main.py:536 - data_simulation()] Waveform/SNR computation timed out (>90s). Skipping event...
```

The *only* parameter fragment ever adjacent to a timeout is the FEW-library `sanity_check_init` line, which carries the mass ratio $q = \mu/M$ alone (not $M$, $e_0$, $p_0$ individually) and only fires when $q$ exceeds the EMRI sanity bound. Adjacency statistics: 333/962 (34.6 %) of seed600 timeouts and 353/1 267 (27.9 %) of seed500 timeouts are immediately preceded by this line, against baseline line frequencies of 17.7 % and 41.9 % of iterations respectively — a $\sim 2\times$ enrichment in seed600 but a *depletion* in seed500, i.e. no reliable parameter proxy. `datamodels/parameter_space.py` contains **no logging calls at all**, and the loop logs nothing between `randomize_parameters()`/`set_host_galaxy_parameters()` and the SNR attempt. **Timeouts cannot be binned by $(M, \mu, e_0, p_0, d_L)$ from any existing log.**

## 6. Minimal instrumentation for the campaign (deliverable c)

A logging-only change (no computed value changes, hence outside the `/physics-change` gate, though it touches a physics-trigger file):

1. **`master_thesis_code/main.py:537`** (SNR-stage catch) — replace the fixed string with one that interpolates `parameter_estimation.parameter_space._parameters_to_dict()` (defined at `datamodels/parameter_space.py:211-227`; already returns all 14 parameters `M, mu, a, p0, e0, x0, luminosity_distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0`), e.g.
   `_ROOT_LOGGER.warning("Waveform/SNR computation timed out (>90s). Skipping event... params=%s", parameter_estimation.parameter_space._parameters_to_dict())`
2. **`master_thesis_code/main.py:569`** (CRB-stage catch) — same interpolation.
3. **`master_thesis_code/main.py:777`** (injection catch) — same interpolation **and** fix the `>90s` → `>30s` mislabel (or interpolate `_TIMEOUT_S`); additionally `sample.redshift` and `redshifted_M` are in scope and should be logged.

Optionally align `main.py:645` (`"Computation exceeded 90s timeout"`) with `_TIMEOUT_S`. Three one-line edits give machine-parseable timeout records (`grep "timed out" | sed`-able dict) sufficient to bin timeout probability by parameter and to reweight or extend the budget where it matters.

## 7. First-order bias assessment (deliverable d)

- **SNR stage (simulation):** timeout rate $0.59\%$–$1.17\%$ of loop iterations (seed600 / seed400–seed500). Crucially, the *number* of timeouts is comparable to the number of detections (seed500: 1 267 timeouts vs 1 551 detections; seed600: 962 vs 2 685). If timed-out events were detection-neutral (detected at the mean rate $\approx 1.3\%$), the lost detections would be only $\sim 0.013 \times 1267 \approx 16$ ($\sim 1\%$ of the sample) — negligible. But timeouts plausibly select *long, harmonic-rich, slowly-evolving* waveforms (low $M$, near-boundary $q$, large initial separation), which at fixed $d_L$ can be *loud*; if their true detection fraction were instead $\mathcal{O}(10\%)$, the loss reaches $\mathcal{O}(8\%)$ of the detected sample with a mass-correlated (hence $M_z$–$d_L$-plane-correlated, hence weakly $H_0$-correlated) profile. Existing logs cannot discriminate these scenarios — that is exactly the G9 instrumentation gap of §6.
- **CRB stage:** $1/1\,646 = 0.061\%$ (seed500) and $1/3\,499 = 0.029\%$ (seed600) of post-threshold events — negligible at first order.
- **Injection stage:** $1.25\%$ of attempted waveforms, uniformly. Because the injection set defines $\widehat{P}_\mathrm{det}(d_L, M_z)$, its timeouts *deplete the denominator* of the Gray et al. (2020) selection normalization in the same slow-waveform region where the simulation *also* drops events — partial cancellation — **but** the budgets differ (30 s vs 90 s, §2 D2), so events with $30\,\mathrm{s} < \tau < 90\,\mathrm{s}$ are kept in the event sample while excluded from $\widehat{P}_\mathrm{det}$, biasing $\widehat{P}_\mathrm{det}$ *low* (and the per-event selection correction *high*) in that parameter region. First-order bound on the affected fraction: $\le 1.25\%$ of injections and $\le 1.2\%$ of simulated iterations; the residual after cancellation is bounded by the budget-asymmetric sliver, i.e. sub-percent of the detected sample — small against the current de-rail effects ($\Delta \mathrm{MAP} \sim 0.1$), but not provably negligible for a percent-level $H_0$ program until binned by parameter.

## VERDICT

**DEVIATION FOUND** (in the logging implementation, not in the physics of the scan itself):

1. **`master_thesis_code/main.py:777`** — injection-stage timeout message states `(>90s)` while the enforced budget is `_TIMEOUT_S = 30` s (`main.py:681`, armed `main.py:732`); the handler string at `main.py:645` is likewise wrong for this path. Log-accuracy defect.
2. **`main.py:475` vs `main.py:732`** — 90 s (simulation) vs 30 s (injection) budget asymmetry means the timeout selection does not cancel between the event sample and $\widehat{P}_\mathrm{det}$; unmodelled selection, bounded at first order by $\sim 1\%$ of attempted events per stage (measured: sim SNR-stage $0.59$–$1.17\%$, injection $1.25\%$, CRB $\le 0.06\%$).
3. **No parameter information at any timeout site** (`main.py:537`, `main.py:569`, `main.py:777`; `parameter_space.py` logs nothing) — timeouts cannot be binned by $(M, \mu, e_0, p_0)$ from existing logs; instrumentation of §6 is required before the campaign.
