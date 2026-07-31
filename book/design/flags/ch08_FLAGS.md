# ch08_FLAGS.md — Chapter 8 ("A Second Handle: the Mass Channel")

Raised by the ch08 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

None of these blocks the chapter. Every one of them is carried on the page in both forms
where the reader can see it, and every disputed number is emitted into the chapter's data
files so the discrepancy stays checkable.

`gen_ch08.py` runs **43 numeric fidelity gates** (`_check`) plus **8 structural guards**
(`raise`) against `CLAIM_2D_BIAS_20260730.md` (C1–C4, C8, C10),
`gate_b_20260730/{c8_reparam,c3c4_allruns,c4_decomposition}_results.json`,
`REALISTIC_READOUT.md` §1, `realistic_scores.csv` and the delivered
`combined_posterior_2d.json`. **All of them
pass**; the items below are the ones that could not be gated, or that needed a definition
pinned down first.

---

## F-ch08-1 — The 2D per-run pull *range* does not reproduce (mean and count do)

- **Cited:** `REALISTIC_READOUT.md` §6 table — 2D "pull vs truth **+3.4 … +4.5 (mean
  +4.04)**", "runs with |pull| > 2: **10/10**".
- **Recomputed** by `gen_ch08.py` from `realistic_scores.csv`, column `pull_2d` — the
  column `score_realistic.py:171` writes as `(map_h_2d − 0.73)/σ_h,2D`:
  - mean **+4.0388** → agrees with +4.04 ✓
  - |pull| > 2 in **10/10** ✓
  - range **+2.474 … +4.735** ✗ (the readout says +3.4 … +4.5)
  - mean 2D MAP **0.807**, bias **+0.077** ✓
- The low end is `seed61000/r2` (MAP 0.78, σ_H0 2.021 → +2.474); the high end is
  `seed61000/r5` (MAP 0.80, σ_H0 1.478 → +4.735). Both are inside the readout's own
  MAP range 0.780–0.820, so the disagreement is in the *pull* row only.
- **Disposition:** the chapter quotes the mean (+4.04), the 10/10 count and the
  **recomputed** range, and says which is which. `ch08_channel.json`
  (`scorecard_summary`) carries the full per-run table so a reader can redo it. No
  reconciliation asserted.

## F-ch08-2 — I8.2's "units dial": the spec's four MAPs are the constant-C sweep, not four unit choices

- **Spec:** `BOOK_DESIGN.md` §1 Ch 8, interactive I8.2 — "units dial (M☉ / 10⁵ M☉ /
  10⁶ M☉ / kg) … MAP walk **0.81329 / 0.78107 / 0.74440 / rail 0.600**"; identically in
  `BOOK_PEDAGOGY.md` Part 4 §Ch 8.
- **Measured** (both re-derived here and present in `c8_reparam_results.json`):
  - those four MAPs are the **constant-C sweep** at C = 1 / 0.3 / 0.1 / ≤0.01
    (`c8_reparam.py` block **[C]**);
  - the four *literal* mass-unit choices are block **[D]**, use a **per-event** scale
    $C_i = M_{z,\det,i}/U$, and measure
    **fraction (code as shipped) 0.81329 · $M_z$ in 10⁶ M☉ 0.80630 · $M_z$ in 10⁵ M☉
    0.85397 · $M_z$ in M☉ 0.86000 (rail)**.
- Attaching the block-[C] numbers to block-[D] labels would be a fabricated pairing, so
  the chapter ships **both dials, each labelled with what it actually is**: an arbitrary
  rescaling constant $C$ (13 steps, the spec's four values among them) and a named-measure
  selector carrying the four measured unit choices. The AHA the spec asks for — a
  published number that moves with an arbitrary choice — is delivered by both, and the
  second one is strictly stronger because the choices are *nameable*.
- **Note for Ch 11 (M5/C8):** if Ch 11 quotes "0.81329 → 0.78107 → 0.74440 → 0.600", it
  must call them a **constant-C walk**, not a unit walk. `README_C8.md` §"Corrections to
  the claim as written", item 1, is explicit that a *consistent* unit change is exactly
  invariant.

## F-ch08-3 — Code line anchors have drifted since the design docs were written

`BOOK_SOURCES_MAP.md` §3 / `BOOK_DESIGN.md` §1 give re-grep anchors that no longer point
at the named code in the current tree. The chapter uses the **spec's** anchors (§3.2:
"copy them from `BOOK_SOURCES_MAP.md`, do not invent"); the current positions are recorded
here so the integrator can re-anchor in one pass:

| object | spec anchor | current tree |
|---|---|---|
| redshift-only candidate filter (1D) | `handler.py:592` (`:593`, `:584-592`) | `handler.py:623` |
| mass window (2D only) | `handler.py:605` (`:594-603`, `:595-604`) | `handler.py:634` |
| `get_possible_hosts_from_ball_tree` | `handler.py:519` | `handler.py:558` |
| `mz_integral` (analytic mass factor) | `bayesian_statistics.py:4363-4370` | `:4442-4459` |
| `single_host_likelihood_batch` | `bayesian_statistics.py:4014` | `:4097` |
| BH-mass denominator inner M integral | `bayesian_statistics.py:3362` | `:3445` |
| `w_G = beta_G / D` | `bayesian_statistics.py:3309-3311` | `:3388-3392` |
| `eddington_shifted_host_mass` | `bayesian_statistics.py:500` | `:500` ✓ |
| production Gaussian mass kernel | `bayesian_statistics.py:3473-3488` | `_mass_trunc_*` at `:537`, `:631`; analytic Gaussian at `:4451` |

The *content* at each site is unchanged — this is line drift, not a semantic conflict.

## F-ch08-4 — Event 606's 2D suppression: "80×" (pedagogy) vs 73.8× measured

- **Cited:** `BOOK_PEDAGOGY.md` Part 4 §Ch 8, I8.3 — "for 606 … an **80×-suppressed** and
  *decreasing* 2D leg".
- **Measured** at $h = 0.73$ from `seed61000/real_r1/diagnostics/event_likelihoods.csv`:
  $\mathcal L^{\rm cat}_{\rm 2D}/\mathcal L^{\rm cat}_{\rm 1D} = 0.013554$, i.e.
  **73.8×**; median over the 41-point grid **72×**. The *direction* statements all hold
  (2D leg argmax 0.600, falling; completion leg argmax 0.860, rising).
- **Disposition:** the chapter quotes the measured **74×** with its chip. The pedagogy's
  "80×" is a design-document round number, not an artifact number.

## F-ch08-5 — Provenance chain of "catalogue σ_lnM ≈ 1.28"

- The chapter quotes σ_lnM ≈ 1.28 (a factor ≈3.6) for the catalogue mass proxy, chipped to
  `CLAIM_2D_BIAS_20260730.md` C4, which states it as `[LOCAL]` support ("P6 work measured
  the mass rejection as strictly one-sided … because σ_Mz/M_z ≈ 1e-4 while catalogue
  σ_lnM ≈ 1.28").
- The originating text is `HANDOFF_20260730.md` §4 — the section that
  `BOOK_SOURCES_MAP.md` §7.2 forbids citing **as current**. That prohibition is about
  HB's *status* (HB was refuted); the catalogue statistic is not part of the refuted
  claim, and the claim file carries it independently.
- **Disposition:** cited to the claim file only; HANDOFF §4 is cited on this page **once**,
  explicitly as historical record, where the chapter tells the HB story.
- Separately: 1.28 and the ratified kernel derivation's **0.58** are *different objects* —
  0.58 is the derivation's floor √(0.553² + 0.184²) (intrinsic + dα), 1.28 is a measured
  median including the propagated stellar-mass and dβ terms. The chapter says which is
  which at every mention, per the notation table's instruction ("state which").

## F-ch08-6 — C10's "39.1%" counts the sign of $(1-w_G)\mathcal L^{\rm comp}$, not of $\mathcal L^{\rm comp}$

- **Cited:** `CLAIM_2D_BIAS_20260730.md` C10 — "only **39.1%** of dark events have a
  positive completion tilt — i.e. `L_comp` pulls DOWN for dark events."
- **Definition, traced:** `gate_b_20260730/attack_c4_decomposition.py:234` computes
  `frac_dark_positive = (dlnC[dark] > 0).mean()` with
  `dlnC = Δ ln[(1−w_G)·L_comp]` — the whole channel-common factor, prefactor included.
  Recomputed: **0.39087** ✓ (exact match).
- Recomputing the same fraction for **$\mathcal L^{\rm comp}$ alone** gives **0.27712** —
  i.e. the claim's conclusion is if anything understated. Both are emitted into
  `ch08_sieve.json` (`c10.dark_frac_positive_completion_tilt`,
  `c10.dark_frac_positive_Lcomp_only`) with the definition spelled out.
- **Disposition:** not a conflict; a definition made explicit. The chapter states the
  39.1% with its definition attached.

## F-ch08-7 — The C4-amended budget's "0.0354 → 0.0061" needed its estimator pinned

`CLAIM_2D_BIAS_20260730.md` C4-amended and `ADJUDICATION_20260730.md` §6.3 both quote
"dark mean catalogue mixture weight **0.0354 → 0.0061** at h = 0.73, a factor 5.8" without
writing the estimator. Recomputed here as the per-event catalogue share of the mixture,

$$\bar w_{\rm cat} = \left\langle \frac{w_G \mathcal L^{\rm cat}}{w_G \mathcal L^{\rm cat} + (1-w_G)\mathcal L^{\rm comp}} \right\rangle_{\rm dark}$$

giving **0.035444 → 0.006125, factor 5.787** — agreement to 4 significant figures in both
channels, so the estimator is identified. Emitted per-h in `ch08_sieve.json`
(`w_cat_dark`), which is what I8.1 plots.

## F-ch08-8 — The CRB table's mass column is $M_z$, and Ch 1's dossier calls it $M$

- `BOOK_DESIGN.md` §1 Ch 1 opens the running-example dossier with
  "EMRI-889 (**M** = 7.25×10⁵ M☉, μ = 10 M☉, d_L = 88.9 Mpc, …)" — the value of the
  `M` column of `prepared_cramer_rao_bounds.csv`.
- That column is the **detector-frame** mass. `gate_b_20260730/c8_reparam.py:59` reads it
  as `M_z = crb["M"]`, and its min/max — 1.33e5 … 1.63e6 M☉, factor 12 — are exactly the
  $M_{z,\det}$ span `README_C8.md` quotes. The notation table distinguishes
  `M` (source-frame) from `Mz` (detector-frame), so the two usages are not interchangeable.
- **Disposition:** Ch 8's dossier row is labelled `M_z` and states the identification with
  its source, rather than silently re-labelling Ch 1's row. `ch08_twofaces.json` names the
  field `M_z_det_Msun`. **For the integrator:** Ch 1's dossier row should say which frame it
  means; the *number* is right either way, because $z \approx 0.02$ for 889 makes the two
  differ by ~2%, below the displayed precision.

---

### Non-flags (checked, no discrepancy)

- C1/C2/C3 class budget, C4-obs counts (64.7% / 32.5% / 1095 / 488 / 487 / 7.78e-3 /
  −504.78 / +0.27), C4-amended partition (487 → +0.236 = 1.49%; 491 → 0.000; 534 → +15.596
  = 98.51%), the dark class profile's argmax move 0.640 → 0.785 with the dark completion
  leg at 0.810, the opposition collapse −24.462 → −0.633, C10's +31.554 / −3.109 /
  −22.718, C8's whole constant-C sweep, the four named measures, `dMAP/dlnC = +0.030909`,
  1D bitwise invariance, and the 3.638e-12-nat reconstruction of the delivered 2D
  posterior — **all reproduce**, several to machine precision.
- The off-r1 replication was recomputed on all ten runs from their own diagnostics CSVs:
  dark channel difference **+15.83 … +17.14, positive in 10/10**; in-cat **−1.83 … +2.97**;
  dark share **84.2 % … 112.5 %**. Consistent with `c3c4_allruns_summary.md`; the chapter
  prints the replicated qualitative claim and never "84%" as the finding
  (`BOOK_SOURCES_MAP.md` §7.5).
