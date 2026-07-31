# ch04_FLAGS.md — Chapter 4 ("The Universe Only Shows You Its Loud Half")

Raised by the ch04 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

Nothing here blocks the chapter. Each item is presented on the page in both forms where it
is visible to the reader.

---

## F-ch04-1 — Injection-pool row count: 200,807 (spec) vs 200,100 (measured) — RESOLVED-BY-ARITHMETIC, still logged

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 4 Sources ("200,807 rows") and
  `BOOK_SOURCES_MAP.md` §5 / §3 S2 ("200,807 rows"), both quoting
  `CLAIM_2D_BIAS_20260730.md:587-588` ("**200,807** rows, z_cut **1.5**").
- **Other spec value:** `BOOK_PEDAGOGY.md` Part 0 ("`injection_pool_mix200k_20260728`,
  **200,100** rows"). The two design documents disagree with each other.
- **Measured by `gen_ch04.py`:** concatenating the 707 `injection_h_0p73_task_*.csv` files
  in `gate_b_20260730/injection_pool_mix200k_20260728/` yields **200,100 data rows**.
- **Arithmetic:** 200,100 data rows + 707 CSV header lines = **200,807 lines**. The two
  figures are the same pool counted two ways (`wc -l` vs `len(df)`).
- **Disposition:** the chapter quotes **200,100 data rows in 707 files** and states the
  header-line identity explicitly in the GW-reader stratum, with both provenance chips.
  Neither number is silently dropped. No reconciliation is asserted beyond the arithmetic,
  which is checkable and is emitted into `ch04_horizon.json`
  (`pool.n_data_rows`, `pool.n_files`, `pool.n_lines_with_headers`).
- **For the integrator / other chapters:** any chapter quoting the pool size should quote
  the data-row count, or say "lines". Ch 6 (I6.2) and Ch 9 (I9.1) both read this pool.

## F-ch04-2 — `dl_max(0.73) = 9.164987 Gpc` is a p_det-grid property, not a pool column

- The fingerprint quoted in `BOOK_DESIGN.md` §1 and `BOOK_SOURCES_MAP.md` §3 S2 is
  reproduced exactly (`SimulationDetectionProbability.get_dl_max(0.73) = 9.1649872`).
- It must **not** be read as the pool's maximum `luminosity_distance` column, which is
  10.686 Gpc. `dl_max` is the survival grid's ceiling: 1.1 × the largest detection horizon
  in the population-measure stratum (8.33181 × 1.1 = 9.16499).
- **Disposition:** no conflict; recorded so the next chapter that quotes the fingerprint
  does not attach it to the wrong quantity.

## F-ch04-3 — Recomputed `D(h)` sits 4.4–6.3% above the production run's own `D(h)`

- **Production values** (authoritative, used by the chapter): the run's own log lines,
  `seed61000/mixture_leg_log_extract.txt`, 41 h-values —
  `D(0.60) = 1.881202e9`, `D(0.73) = 1.520637e9`, `D(0.86) = 1.257878e9` Mpc³ sr⁻¹.
- **Fresh recomputation** via `precompute_completion_denominator(...)` on the staged pool:
  4.44–6.27% higher, ratio drifting smoothly with h.
- **Cause (identified, not a defect):** production passes a `CompletenessModel`, which
  switches `D(h)` onto the **sky-aware** branch — `p_det` per ecliptic-latitude band,
  weighted by equal-area HEALPix pixel counts (`bayesian_statistics.py:1074-1088`,
  Change 2). The plain call uses the pooled isotropic survival. The `z_max_cap` argument is
  very nearly a no-op here: the p_det horizon's own `z_max(h)` binds at every grid point
  except the top one, where it would exceed the analysis cap and `z_max(0.86) = 1.5` takes
  over (run log; issue #30 selection-domain cap).
- **Disposition:** the widget plots the recomputed integrand as a **shape** and quotes the
  production `D(h)` as the **number**, states the 1.0444–1.0627 level factor on the page,
  and notes that the factor itself varies by only **1.75%** across the grid (so the shape
  is the same object). Both arrays ship in `ch04_horizon.json`
  (`D_h_production`, `D_h_pooled_recompute`) so a reader can check.

## F-ch04-4 — `BOOK_DESIGN.md` §1's I4.1 static fallback reads "MAPs 0.60 / 0.60 / 0.73"; this run measures 0.600 / (recorded 0.60) / 0.740

- **Spec:** Ch 4 interactive table, I4.1 fallback — "static: three posteriors, MAPs
  0.60 / 0.60 / 0.73 annotated".
- **Measured, live, on the specified data source** (`seed61000/real_r1`, 1588 events):
  denominator deleted → MAP **0.600** (mean 0.600000); full-volume `D(h)` → MAP **0.740**
  (mean **0.7321**).
- **Assessment — not a contradiction, a venue difference.** The `0.60 → 0.73` pair is the
  **Phase 32** measurement (`ledger #9`, `H0R:1980`) on that era's venue; the spec's own
  source line for I4.1 says the middle state is a *recorded overlay* of exactly those
  Phase-32 numbers. The live `0.740 / 0.7321` reproduces `REALISTIC_READOUT.md` §1's
  published row for seed 61000 / r1 **to the digit**, and the generator enforces that as a
  hard gate (it raises rather than writing a file if it ever stops matching).
- **Disposition:** the page shows both, labelled and separated — the bias rail carries the
  Phase-32 `−0.178 → 0.000` pair (as `BOOK_DESIGN.md` §1 mandates, with a venue note in the
  tooltip), and an adjudicator-voice block states explicitly that the live `0.740` is one
  realization and is **not** a bias measurement. The static fallback in `<noscript>` carries
  all three numbers with their venues. No number was adjusted.

## F-ch04-5 — Build portability: Ch 4's specified data sources are partly untracked

- `BOOK_DESIGN.md` §1 and `BOOK_PEDAGOGY.md` Part 0 route the majority of the book's
  interactives through `real_r*/diagnostics/event_likelihoods.csv` and the injection pool.
  **Neither is git-tracked** — both exist only in the working tree of the main checkout, so
  neither is present in this worktree or in a fresh CI clone. (`git ls-files` finds 817
  tracked files under `results/campaign51_20260728/`; the diagnostics CSVs and the 707 pool
  CSVs are not among them, and they are not `.gitignore`d either — simply never committed.)
- **Disposition for Ch 4:** I4.1 was re-based onto **tracked** artifacts only —
  `real_r1/posteriors/h_0_*.json` (via the project's own `load_posterior_jsons` /
  `build_likelihood_array`) plus `mixture_leg_log_extract.txt` for `D(h)` — and reproduces
  the diagnostics-CSV route to 3e-15 in the posterior mean. I4.2 needs the pool, so
  `gen_ch04.py` resolves it from this repo root, then from a sibling `MasterThesisCode`
  checkout, and if neither exists it **keeps the committed `ch04_horizon.json` and prints a
  NOTICE** rather than failing the build or writing a degraded file.
- **For the integrator:** wave-2/3 chapters that plan to read `event_likelihoods.csv` will
  hit this. Either those artifacts get committed, or every generator needs the same
  tracked-first / sibling-fallback / keep-committed-output pattern.
