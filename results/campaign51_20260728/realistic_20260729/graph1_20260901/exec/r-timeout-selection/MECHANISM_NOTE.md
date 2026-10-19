# MECHANISM_NOTE — how waveform timeouts enter (and do not enter) `p_det`

Node: r-timeout-selection (batch 2, prereg author C). Date 2026-09-04. Read-only code trace; every
claim below is `[LOCAL]` (quoted from the working tree at HEAD `79c44608`, or counted from the
locally fetched seed61000 logs) unless tagged otherwise. No registered statistic was computed.

## 1. The three catch sites (working-tree line numbers)

**Simulation loop, SNR stage** — `darksiren_emri/main.py:471-474` installs the alarm handler
(`raise TimeoutError("Computation exceeded the alarm timeout")`); `main.py:619` arms
`signal.alarm(90)`; the catch at `main.py:763-771`:
```
        except TimeoutError:
            skip_counts["snr:TimeoutError"] = skip_counts.get("snr:TimeoutError", 0) + 1
            # G9 gate: log the full parameter set so timeout selection can be
            # binned by (M, mu, e0, p0, ...) — see .planning/gate/G9_timeout_scan.md
            _ROOT_LOGGER.warning(
                "Waveform/SNR computation timed out (>90s). Skipping event... params=%s",
                parameter_estimation.parameter_space._parameters_to_dict(),
            )
            continue
```
`continue` fires BEFORE `passed = snr >= cosmological_model.snr_threshold` (`main.py:778`) and before
any CRB write: a timed-out draw is neither a detection nor a non-detection — it is not an event.

**Simulation loop, CRB stage** — `main.py:792` `signal.alarm(90)`; catch `main.py:812-818`, same
pattern (`crb:TimeoutError`, params logged, `continue`). n = 2 in seed61000 — negligible.

**Injection campaign (the p_det pool)** — `main.py:1094-1099`:
```
    # Aligned with the main simulation loop's 90 s alarm (readiness sweep A1,
    # 2026-07-03): the injection SNR uses the FULL T-yr generator and depth 1.5
    # lifts M_z into corners never timing-profiled at the old 30 s budget;
    # timed-out events are DROPPED from the pool, so a timeout-rate correlation
    # with (d_L, M_z) would bias the p_det grid. Smoke test bins the counter.
    _TIMEOUT_S = 90
```
armed at `main.py:1225` (`signal.alarm(_TIMEOUT_S)`); catch `main.py:1293-1302`:
```
        except TimeoutError:
            # G9 gate: params logged for timeout binning (smoke test checks
            # for (d_L, M_z) correlation before full-campaign sizing).
            timeout_count += 1
            _ROOT_LOGGER.warning(
                "Injection waveform/SNR computation timed out (>%ss, %d total). "
                "Skipping event... params=%s", ...)
            continue
```
`continue` fires BEFORE `results.append({... "SNR": float(snr), ...})` at `main.py:1308-1333`
("Store ALL events regardless of SNR (per D-03: do NOT threshold)"). **A timed-out injection draw
is ABSENT from the pool CSV** — not a NaN, not a flag, not an SNR=0 row. Verified on the pool of
record (`gate_b_20260730/injection_pool_mix200k_20260728`, 707 files, 200,100 rows): `SNR` has 0
NaN, 14 rows with `SNR == 0.0` (legitimate near-zero SNR, not sentinels), and the only NaNs are
`t_plunge_yr`/`p0` on the 6,000 rows written at commit `a9f29e82` (tasks 0-17, before the p0
provenance columns existed — `code_rev` column). The final tally line
(`main.py:1349-1352`, `"... {timeout_count} timeouts @ {_TIMEOUT_S}s)"`) lives only in the pool
build's cluster log, which is NOT in the local tree (`g-population` gate, REGISTRATION_DRAFT §6).

The G7 row 8 premise "30 s inj / 90 s sim asymmetry" is STALE: both budgets are 90 s since the
2026-07-03 alignment (`main.py:1094-1099`; the pool of record was built 2026-07-28 at
`f6449051`/`a9f29e82`, after the alignment). A `git show 3273fa59` confirms `_TIMEOUT_S = 90`
entered before the pool build. Remaining asymmetry is hardware/load, not the constant (§4).

## 2. How the survival estimator consumes the pool

`darksiren_emri/bayesian_inference/simulation_detection_probability.py:327` `dfs.append(pd.read_csv(f))`,
`:337` `pd.concat(dfs)`, `:350` `required_cols = {"z", "M", "SNR", "h_inj", "luminosity_distance", "qS"}`,
`:415` `snr_full = self._pooled_df["SNR"].values`, `:490`
`self._d_hor = self._snr_raw * self._dl_raw / self._snr_threshold`, `:509-513` sorted horizons;
`p_det(d_L) = P(d_hor >= d_L)` (`:17`, `:484-485`). There is no `dropna`, no timeout column, no
weight for missing draws: **the denominator of every survival is the number of COMPLETED draws.**
Timed-out draws are therefore dropped from BOTH numerator and denominator of `p_det` — they are
not "counted as non-detections" (that would require a row with `d_hor = 0`, which the loop never
writes). Consequence: every pool-built object is conditional on "waveform completed in 90 s":
`S(d_L | completed)`, `S(d_L | M_z, completed)`, `D(h)` (`bayesian_statistics.py:1170-1324`,
`precompute_completion_denominator`, integrand `p_det · dVc/(1+z)` at `:1286-1300`).

## 3. What the simulation side does with the same population

The kept event set is `SNR >= 20 ∧ completed(SNR stage) ∧ completed(CRB stage) ∧ D1 p0-window
∧ CRB numerics`. The D1 window is NOT a timeout: `datamodels/parameter_space.py:110-111`
(`lower_limit=10.0, upper_limit=16.0`, retained "SNAPSHOT-mode bounds only" per `:99-108`) is
still the bound the 5-point stencil checks at `parameter_estimation/parameter_estimation.py:271-276`
(`raise ParameterOutOfBoundsError("Tried to set parameter to value out of bounds in derivative.")`),
caught at `main.py:793-800` with a bare string (no params). Production p0 is the plunge-window
root (`plunge_window.py:108-111`, no upper clamp), so every SNR-passer with p0 ∉ [10.002, 15.998]
dies here silently. `[LOCAL]` counts, `seed61000/cluster_logs_fetch_20260904/logs/simulate_6088772_*.err`
(100/100 tasks, manifest md5 `ebf09fc4ab66b55e4eb592731ee46ae6`):

| stage / outcome | count | note |
|---|---:|---|
| loop iterations (Σ last "X / Y evaluations successful", 100 tasks) | **89,456** | Y per task |
| SNR threshold check failed | 78,841 | INFO line, **no params** |
| SNR threshold check successful | 5,921 | |
| SNR-stage `timed out` (params logged) | 820 | |
| CRB-stage `timed out` (params logged) | 2 | |
| ZeroDivisionError (SNR stage) | 3,488 | no params |
| CRB `in dervative` = D1 p0-window drops | **4,071** | = 68.8 % of SNR-passers; no params |
| CRB other (ZeroDiv/Runtime/Value) | 39 | params logged |
| "Mass ratio" lines | 77,936 | FEW `sanity_check_init` INFO lines, not skips (78,841+5,921+822+3,488 = 89,072 ≈ Y) |
| CRB rows written (Σ X) / in `prepared_cramer_rao_bounds.csv` | 1,788 / 1,590 | walltime-cancel flush loss + prep filters, disclosed |

Kept p0 range in the CSV: **[10.0025, 15.987]** — the D1 window, exactly. Kept M range
[1.33e5, 1.63e6] M☉.

**Two framing corrections that follow mechanically (no ruling):**
(a) The per-DRAW SNR-stage timeout rate is **822 / 89,456 = 0.92 %**. The "34 %" of rows #342/#355
is `820 / (1590 + 820 + 2)` — conditional on {kept ∪ timeout}, as the read's own gap 4 discloses.
The two are different objects; only the per-draw rate is the one `p_det` cares about.
(b) The p0-axis read ("100 % timeout above p0 ≈ 20", 13.6σ) is **not evaluable from these logs**:
the denominator {kept ∪ timeout} contains NO non-timeout draw with p0 > 16 by construction (SNR-
passers with p0 > 16 die at the D1 gate without params; SNR-failers log no params). Bin 4 is
therefore 100 % by construction, not by measurement. The pool of record contains 46.6 % of its
a-stratum rows at p0 > 16 (max 107.5) and 21.2 % at p0 < 10 — waveforms at those p0 do complete
in 90 s on the pool's hardware. The same construction affects the M axis less severely (kept M is
not gated) but the M-bin rates are likewise conditional on {kept ∪ timeout}, i.e. they mix the
timeout rate with the SNR-pass rate per bin; the per-draw M-binned rate needs the pool as the
completed-draw denominator (REGISTRATION_DRAFT §4, S1.2, with the `g-closure` sum).

## 4. What could still make p_det blind to the truncation

If the pool's timeout process equals the simulation's (same 90 s, same generator `T = 4.5 yr`,
`dt = 10 s` — CRB CSV columns `T`, `dt`; same emcee M1 draw; same z-depth 1.5), the truncation is a
**shared filter** (the D1 "coverage-invisible" class): `p_det` is the correct selection function
of the completed population, and the only unmodelled term is the HOST-CONDITIONED mass prior
`p_M(M)` inside the with-BH completion leg (`bayesian_statistics.py:871-950`,
`g(z) = ∫ p_det(d_L(z), M(1+z)) p_M(M) dM`, `p_M` = truncated lognormal × R_eff about the host BH mass, `:807`, `:839-841`; consumers `completion_mass_factor_g` `:2143`, `_g_sel` `:2276`, and the catalogue-leg `mz` expectation `:814`), which integrate
an untruncated host-mass prior against a completed-conditional `p_det` (the `emri_rate` measure in `D(h)` is mass-integrated and z-only, `:1292-1297`). Two things break the sharing:
(i) **hardware** — `cluster/inject.sbatch:17` `--partition=gpu_a100_short`; `cluster/simulate.sbatch:36`
`gpu_h100_short,gpu_h100_il,gpu_a100_short,gpu_a100_il` (the seed61000 node list `uc2n561…579` is
in the `.out` logs; GPU type per node is a cluster read, `g-hardware`); a wall-clock cut on faster
hardware is a different cut in parameter space; (ii) **load** — 100 concurrent SNR-only tasks vs
CRB tasks. Whether (i)/(ii) matter is exactly Q1's S1.2.

## 5. Honest physical hypothesis for the timeouts (design input, not a finding)

Kept events generate in `generation_time` 0.16–0.62 s (CRB CSV column, N = 1590). A 90 s timeout
is therefore a >150× slowdown — a distinct regime, not the tail of one distribution. Candidate
drivers, each with the code reference: (1) low M → orbital frequency ∝ 1/M → more radial/azimuthal
cycles and harmonics inside the fixed 4.5-yr, `dt = 10 s` span (the FEW trajectory integrator and
mode sum scale with cycles); (2) at fixed `t_plunge ~ U[0, 4.5 yr]` the plunge-window root gives
LARGER p0 at LOWER M (`docs/derivations/plunge_window_initial_conditions.md:189`: at M_z ≈ 1e6
p0 spans (1.9, 10.83]; the low-M draws sit at p0 ≫ 16), i.e. the M and p0 axes of the read are one
coupled axis; (3) p0 far above FEW's documented Pn5AAK input domain [10, 16]
(`parameter_space.py:99-101`) may put the trajectory root-find/flux interpolation outside its
tuned grid. `Refute by:` the rescue re-run of the 822 logged parameter sets at a 600 s budget
(REGISTRATION_DRAFT §8, NOT-covered): a bimodal completion-time distribution with the mass ratio
`q = mu/M` or p0 as the separator falsifies (1); completion of most draws in < 300 s falsifies the
"pathological" reading and makes the budget a runtime fix. The timeout constant is a runtime
constant in `main.py` (`signal.alarm(90)` at `:619`/`:792`, `_TIMEOUT_S = 90` at `:1099`) — NOT in
`constants.py`, NOT a physics-trigger value; changing it is `instrumentation`, but re-simulating a
pool or a campaign under it is a NOT-covered [DO] (REGISTRATION_DRAFT §8).
