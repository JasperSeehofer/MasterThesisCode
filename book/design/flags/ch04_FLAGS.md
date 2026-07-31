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

---

# REVISION 2026-07-31 — post-review pass (`REVISION_WORKLIST.md` §C-ch04)

Appended, not rewritten: F-ch04-1 … F-ch04-5 above are the record as it stood at build time
and none of them changed status. This section records what this pass changed, and opens one
new flag.

## F-ch04-6 — NEW / RESOLVED-BY-MANDATE: the dossier's σ_dL row carried an absolute Gpc value under a fractional label

- **What was wrong.** The dossier at `ch04-loud-half.html:627` printed
  `88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)`, and `gen_ch04.py` shipped the same slip in the data:
  `event889.sigma_dL_over_dL = 7.98e-05`. That number is √(CRB variance) in **Gpc** — the
  absolute σ_dL — not a fraction. The fractional precision is
  7.98×10⁻⁵ Gpc / 0.0888792 Gpc = **8.98×10⁻⁴** (×11.25 slip).
- **Adjudication.** `REVISION_WORKLIST.md` §A-D1 (author mandate): the measured value is
  adopted book-wide, chapters stop printing both, and each affected page carries a one-line
  erratum. Ch 4 was one of the three dossier sites named by expert A (B2).
- **What this pass did.**
  - Dossier row → the canonical string `d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴`.
  - Canonical erratum note added immediately under the dossier (a `.ch04-note`, not a boxed
    OPEN dispute), pointing at ch01 flag F1 / BUILD_REPORT §5.1 item 1.
  - **Key rename in `gen_ch04.py`** so the slip cannot recur silently:
    `event889.sigma_dL_Gpc = 7.98e-05` (what the CRB column actually holds) **and** a new
    `event889.sigma_dL_over_dL = 8.98e-04` (computed as σ/d_L). The comment at the
    computation site states the units explicitly.
- **Status:** RESOLVED. `qa_gates.py` gate D1 passes on ch04: the string `8.0×10⁻⁵` survives
  on the page only inside the erratum sentence.

## Worklist items, dispositioned

- **[P1] tomas M2 — what `p_det` is a function of.** §2 gains a paragraph after the
  h-invariance argument: `p_det` here is a function of distance alone, i.e. a **marginal**
  over the injection population's **intrinsic** parameters; that marginalisation is exact
  only if the events it is applied to are drawn from the same population; Chapter 9 measures
  a case where they are not, with a `⏭ Ch 9` chip. Phenomenon + chapter only — the C9 verdict
  and its numbers stay in Ch 9 (D4).
- **[P1] mara MAJOR-5 — Q4.3 / Q4.4 rungs.** Q4.3's second mechanism no longer invokes
  "effective sample size" or "minimum ESS per node" (campaign-design vocabulary that appears
  nowhere in ch00–ch04); it is now **support mismatch**, argued from objects this chapter
  owns: a thin band in the pool ⇒ `p_det` reconstructed from a handful of horizons ⇒ a fixed
  tilt on every event overlapping that band ⇒ multiplied N times because `D(h)` is shared,
  exactly like §3's 639 nats. Ledger #8 is named as the extreme form of the same failure, so
  the two mechanisms stay distinct. Q4.4 now ends at "…*is* the selection correction" plus one
  sentence that a claimed cancellation is a theorem to be tested, with a bare `⏭ Ch 9` chip;
  Σ_glob ≡ n̄β_G, the −17.2% residual, claim C9 and the `G1_beta_g_check.md` anchor are all
  removed from the answer body. Both answers are now derivable from ch00–ch04 alone.
- **[P1] ped M6 (receiver) — Ch 2's β(h)^N box lands in §4.** §4 "Counted exactly once" now
  carries the plain-product derivation (`posterior_combination.py:284-330`, Loredo 2004,
  Mandel/Farr/Gair 2019 §3) with a one-sentence hand-off lead-in ("Chapter 2 promised the
  justification here, because it is about D(h)"), placed immediately before the existing
  ledger-#20 double-count receipt — so the theorem and its violation now sit together at the
  rung that owns `D(h)`. Provenance panel gains the matching convention line.
  **For the ch02 agent:** the box is received here; ch02 should keep only the one-sentence
  column statement + `⏭ Ch 4` chip, per ped M6. If ch02 instead keeps its copy, this becomes a
  duplicate and one of the two must go — flag it to the integrator rather than deleting
  either silently.
- **[P2] mara MINOR-3 — guess-marker desync.** The hand-rolled predict path is replaced by
  `Book.predictValue` (the helper built for R-ch04-1), and the slider is **disabled on lock**
  (ch00's `setLocked` pattern). One code path now owns the stored value, the readout and the
  drawn marker; dragging after locking is impossible, so they cannot disagree. The
  `data-predict` attribute is kept in sync as a print/no-JS fallback. Persistence key is
  unchanged (`book-predict:ch04-map-guess`), so Ch 11's re-surfacing still works.
- **[P2] synth — §5's zero-candidate framing consumes Ch 3's regenerated census.** Ch 4 never
  printed 552 (grep: zero hits before this pass), so there was no wrong number to correct;
  what was missing was the number itself. §5 now states **607 of the 1590 rows** have no
  catalogue galaxy in the ball at the production 1.5σ radius, chipped to the ch03 census, and
  keeps expert A's P7 discipline verbatim in force: it is a **reconstruction** count on the
  truth catalogue, *never a drop count* — supported on-page by this run's own gate, which
  reports zero non-positive likelihood cells (`ch04_denominator.json.checks
  .n_nonpositive_cells = 0`, enforced as a raise in `gen_ch04.py`). The beat that §5 sets up
  (what does the estimator *say* about an empty ball?) is left unanswered for Ch 5.
- **[D4] one spoiler softened.** The adjudicator block's closing forward reference said the 1D
  channel's apparent unbiasedness "is later shown to be a near-cancellation of opposing terms"
  — a Ch 11 verdict stated in Ch 4. Reworded to name the question, not the answer. Expert A's
  P7-praised sentences (the Phase-32 / r1 venue separation and "that is not a bias
  measurement…") are untouched.

## Not done, and why

- **BW3 inline chips (§D-5).** I4.1's mode buttons already carry `data-hypothesis="9"`, and
  the `local` verdict text already hard-codes "Has this been tried? Yes — it is ledger row
  #9". Under §B-4 that is the case for `data-hypothesis-verdict="inline"`, but §D-5's own
  acceptance criterion names this exact widget as the demo that *should* volunteer the chip.
  The mechanism is the integrator's (§D-5) and the two readings conflict, so ch04 changed
  nothing and leaves the call to integrator pass 2.
- **Rail rows (§D-4, ped M7).** `renderRail` still emits ch04's two rows locally; the
  cumulative `BOOK_BIAS_ROWS` mechanism is the integrator's.

## Reproduction

`gen_ch04.py` re-run clean after the key rename: 1588/1588 events, MAP on 0.74 / off 0.6,
mean on 0.732059, `REALISTIC_READOUT.md §1` gate PASS; pool 200,100 rows in 707 files,
dl_max h-invariant at 9.1649872 Gpc for h = 0.60/0.73/0.86, survival max|Δ| = 0.0,
D pooled/production 1.0444–1.0627. No physics number on the page changed.
