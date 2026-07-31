# ch10_FLAGS.md — Chapter 10 ("Is It Calibrated?")

Raised by the ch10 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

Nothing here blocks the chapter. Every item is presented **on the page in both forms**,
and both forms are emitted into the chapter's data files so a reviewer can check either.

---

## F-ch10-1 — C11's two quoted bias bands: lower endpoints reproduce exactly, upper endpoints do not — **OPEN**

- **Spec / cited value.** `CLAIM_2D_BIAS_20260730.md` C11 (and, verbatim,
  `gate_b_20260730/ADJUDICATION_20260730.md` §1): *"pp_coverage extension to comp_frac
  0.008–0.234 … bias **+0.0008..+0.0097** at comp_frac 0.06–0.09 and **+0.0034..+0.0181**
  at 0.13–0.24 … **6–16×** below +0.077."* Carried into `BOOK_SOURCES_MAP.md` §3 X4 and
  `BOOK_DESIGN.md` §1 Ch 10.
- **Measured by `gen_ch10.py`.** Re-running the archived harness cells that span exactly
  that completion-fraction window — `results/pp_coverage_deepvenue_20260730/`
  (`z_support` 0.38/0.39/0.41/0.43) plus `results/pp_coverage_deepvenue_20260710/`
  (`z_support` 0.2/0.3/0.5/1.0), all `kernel="volume"`, `mixture_mode="two_branch"`,
  σ_z ∈ {0.015, 0.035}, truths {0.62, 0.72, 0.84} — gives

  | band | claim | recomputed | agreement |
  |---|---|---|---|
  | comp_frac 0.06–0.09 | +0.0008 … +0.0097 | **+0.0008 … +0.0078** | lower endpoint exact; upper differs |
  | comp_frac 0.13–0.24 | +0.0034 … +0.0181 | **+0.0034 … +0.0157** | lower endpoint exact; upper differs |

  The window endpoints themselves reproduce exactly (measured comp_frac range
  0.00847–0.2337 vs the claim's "0.008–0.234").
- **Strength of the recomputation.** This is not an independent re-implementation: the
  generator re-runs each cell **from its own archived `config` block**, with the archived
  `seed`, and asserts **bit-equality** of `coverage`, `map_bias`, `rail_fraction` and
  `completion_fraction` against the stored JSON before writing anything. All 16 cells ×
  3 truths pass. The archives' own `.log` files carry the same values
  (e.g. `pp_zs0.38_sz0.035_volume.log`: `h_true=0.7200 … bias=+0.0078`;
  `h_true=0.8400 … bias=+0.0157`). So the recomputed numbers *are* the archive's numbers.
- **Where the claim's upper endpoints may come from.** `+0.00963` at comp_frac 0.0847
  does exist — in `results/pp_fullpower_20260727/pp_cat_lcat_zs0.43_sky1e-4_h0.84.json`,
  which is a **different harness family** (`catalogue_mode=True`, `mixture_mode="lcat"`,
  the impostor-ball universe, n_realizations 2000). If C11's band pools the continuum
  `two_branch` cells with the catalogue-mode cells, `+0.0097` is accounted for. A scan of
  every `results/pp_*/**.json` found **no** archived cell reproducing `+0.0181` inside
  comp_frac 0.13–0.24 (nearest: `pp_cat_lcat_zs0.30_sky2e-4_h0.62`, comp_frac 0.2142,
  bias +0.0174 — again catalogue-mode, and a 120-event cell).
- **Does it change the verdict?** No, in either direction. C11's conclusion is that the
  completion leg's calibration is far too small to own the +0.077 2D bias. The recomputed
  maximum (+0.0157) is *smaller* than the claimed one (+0.0181), so the exoneration is if
  anything stronger. **This flag is about provenance, not about the verdict.**
- **Disposition on the page.** §4 quotes C11's numbers verbatim in the adjudicator's voice
  with its badge and chip, then shows the archive-gated recomputation immediately beside
  it and names the disagreement as a disagreement. `data/ch10_pp.json`
  (`c11_window.bands`, `c11_window.band_disagreement`) carries both, plus the pointer to
  the `pp_fullpower_20260727` candidate. **Nothing is reconciled.**
- **For the author / integrator.** Worth an explicit note in the claim file recording which
  harness families the C11 band pools, and re-deriving the "6–16×" ratio from the named
  cells. As stated, "6–16×" corresponds to a bias band of roughly [0.0048, 0.0128], which
  is neither of the two quoted bands.

---

## F-ch10-2 — "the 3 loudest carry 46%": denominator unstated — **AMBIGUITY, both carried**

- **Spec / cited value.** `IDEALIZED_BASELINE_READOUT.md:42-47` and
  `idealization_audit/IDEALIZATION_LEDGER.md` §1: *"The **3 loudest** (SNR 995–1425,
  z ≈ 0.016–0.021) carry **46%** of the total information by themselves."* Carried into
  `BOOK_SOURCES_MAP.md` §3 R1 and `BOOK_DESIGN.md` §1 Ch 10.
- **Measured by `gen_ch10.py`** on the canonical `run_seed61000/posteriors_fixed`, using
  the audit script's own statistic (signed 3-point second difference of Σᵢ ln Lᵢ at
  h ∈ {0.725, 0.730, 0.735}):
  - in-catalogue curvature **+241.335**, dark **−3.003**, signed total **+238.332**
    (ledger: 241.3 / −3.0 / 101% / −1% — **exact match**, gated);
  - σ_h = 3.24×10⁻⁴ → **σ_H0 = 0.0324 km/s/Mpc** (ledger: 0.032 — **match**, gated);
  - the three loudest in-catalogue events (889, 1536, 118; SNR 1425 / 1068 / 995) sum to
    **112.006**, which is **46.41 %** of the in-catalogue curvature and **47.00 %** of the
    signed total.
- **The ambiguity.** The ledger writes "46% of the total", but 46% is the *in-catalogue*
  share; the *total* share is 47.0%. Since the same paragraph reports the in-catalogue
  share as "101% of the total", the denominator convention is not uniform in the source.
- **Disposition.** The page quotes the ledger's "46%" with its chip and immediately gives
  both recomputed denominators. `data/ch10_closure.json` (`golden3.share_of_in_catalog`,
  `golden3.share_of_total`, `golden3.ledger_quotes`) carries all three numbers. No
  substitution is made.

---

## F-ch10-3 — I10.1's card label says "run 200 universes"; the archived ensembles are 120 — **LABEL, corrected on the page**

- **Spec value.** `BOOK_DESIGN.md` §1 Ch 10 and `BOOK_PEDAGOGY.md` Part 4 both describe
  I10.1's control as *"press 'run 200 universes' (precomputed grid)"*.
- **Measured.** Every archived cell used by the widget has `n_realizations = 120`
  (`pp_coverage_deepvenue_20260710`, `_20260730`). The 500-realization archives
  (`pp_coverage_absolute_20260726`) are `mixture_mode="absolute"` cells — a different
  estimator, not the two_branch ladder C11 rests on — and the 2000-realization ones are
  the catalogue-mode family.
- **Disposition.** The widget's button reads the count out of the data
  (`cells[].n_realizations`) and says **"run the 120-universe ensemble"**. The card's
  round number is not carried into the page. Binomial standard errors on the coverage
  numbers (±0.043 at 68% for n = 120) are shown next to them so the reader is not invited
  to over-read a third digit.

---

## Non-flags (checked, consistent — recorded so a reviewer need not re-check)

- σ→0 byte-identity md5s `1e81ba22` (1D) / `733c8d32` (2D): quoted verbatim from
  `HANDOFF_20260730.md` §1 and `REALISTIC_READOUT.md` §2 P5; not recomputed (the control
  posteriors are cluster-side). Presented as a **recorded** measurement, chipped.
- The 0.67 closure (MAP 0.670, 1343 events, σ_h = 4.42e-4, peak 0.670053, +0.12σ) is
  quoted from `HANDOFF_20260730.md:15-23`, **with** that source's own caveats (fitted bins
  sit 11.3σ out; the GPU array timed out at 1345 vs the baselines' 1590 detections).
- The n-scaling ladder (cov68 0.63 → 0.38 → 0.12 at h_true = 0.72) is read straight from
  the archives: 0.6333 (`pp_coverage_exactmode_20260711/pp_exact_zs0.3_sz0.035.json`,
  n = 250), 0.3833 and 0.1250 (`pp_coverage_noisemodel_20260711/pp_nscale_constsig_n{1000,4000}`).
  Matches `pp_coverage_noisemodel_20260711/SUMMARY.md:80-86`.
- `sig0_control` is **not** used anywhere in this chapter (sources map §7.6: it carries the
  `generator_marginal` estimand). It is *named* in §5 only as the object Gate A1 read to
  confirm C6, which is exactly what the claim file does.
- `REALISTIC_READOUT.md` §6's struck-out sentence "the 1D channel is the defensible one"
  is **not** used (sources map §7.3). §5 of the chapter follows C5 and the readout's own
  2026-07-30 amendment.
