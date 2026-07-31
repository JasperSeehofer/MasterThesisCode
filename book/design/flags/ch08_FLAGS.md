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

---

# REVISION — 2026-07-31 (append-only; nothing above this line was edited)

Applied by the ch08 revision agent against `book/design/REVISION_WORKLIST.md` §C-ch08
(+ the book-wide decisions D1/D3/D4/D5 in §A). The items below are new flags and
recorded decisions; the original flags F-ch08-1 … F-ch08-8 stand as written.

## F-ch08-9 — The cell-B C4 partition is recomputed for the book, and is in no adjudicated artifact  ⚑ FOR THE AUTHOR

- **Where it is used:** §5's C4 block, §5's relocated deletion trap, Q8.3's answer, I8.1's
  Panel-B verdict strings, and the provenance panel — every one of them marked
  *"recomputed for this book"*.
- **What is adjudicated** (`CELLB_READOUT_20260731.md`, evaluate 6103219 / combine 6103220,
  code `7fd60bb`): cell B's MAPs (1D 0.7450, 2D 0.7900), the per-class channel differences
  (dark **+18.00**, in-cat **−1.80**, total **+16.20**), and `w_G(h)`'s bit-identity with the
  #53 curve. `gen_ch08.py` reproduces all of these from cell B's own diagnostics CSV
  (recomputed: +17.9958 / −1.7984 / +16.1974; w_G 0.16251748 / 0.12150388 / 0.10387316 at
  h = 0.60/0.73/0.81).
- **What is NOT adjudicated** — first computed by the book's expert review B (§0 of
  `book/design/reviews/expert_B_ch07-11_cellB.md`), reproduced here independently:

  | quantity | r1 (scattered) | cell B (unscattered) |
  |---|---|---|
  | events with a live 1D catalogue leg anywhere | 1095 | **982** |
  | of those, 2D-zero at every h | 488 (487 dark) | **688 (all 688 dark)** |
  | dark channel diff carried by those | +0.24 = **1.5%** | +3.46 = **19.2%** |
  | …by the both-dead group | 0.00 (n = 491) | 0.00 (n = 605) |
  | …by the survivors | +15.60 = **98.5%** (n = 534) | +14.53 = **80.8%** (n = 219) |
  | dark mean catalogue mixture weight 1D → 2D | 0.0354 → 0.0061 (×5.79) | **0.0361 → 0.0043 (×8.39)** |
  | dark fraction with `L_cat_with_bh == 0` at h = 0.73 | 0.647 | **0.855** |

- **Two small departures from the review's transcription**, both in the book's favour and
  both measured here: (a) the review writes the zeroed count as "688 (687)" by analogy with
  r1's "488 (487)" — recomputed, **all 688 are dark**, so there is no in-catalogue member to
  put in parentheses; (b) the review's survivors share is quoted as 80.7%; recomputed it is
  **80.763%**, which the chapter prints as **80.8%** (the pair then sums to 100.0). The
  generator gates against the review's one-decimal figures at tol 0.1.
- **Disposition:** used, with a `recomputed for this book` provenance chip at every site.
  **Author's call** (worklist §F item 4) whether this partition is promoted into an
  adjudicated artifact or a `CLAIM_2D_BIAS_20260730.md` C4 amendment. The book asserts only
  the structural claim that survives on both configurations — *deletion is the minority
  carrier, de-weighting the majority one* — and explicitly refuses to print 98.5% as the
  mechanism's signature (expB MJ-4's acceptance criterion).
- Emitted as `ch08_sieve.json → cell_b` (with `provenance`), gated by 15 new `_check`s plus
  two structural guards in `gen_ch08.py` (`build_cellb`): the dark channel difference must be
  positive, and the zeroed share must stay below the survivors' share — if either flips, the
  sentence in §5 is false and the build stops.

## F-ch08-10 — σ_Mz/M_z is a both-values pair, carried at five sites (was: five unflagged assertions of 10⁻⁴)

- **The conflict** (tomas B3; raised *for this chapter* by `ch06_FLAGS.md` F-ch06-5):
  `CLAIM_2D_BIAS_20260730.md:172` (C4) states σ_Mz/M_z ≈ **10⁻⁴**; the same stored quantity —
  `sqrt(delta_M_delta_M)/M` of `prepared_cramer_rao_bounds.csv` — measures a **median
  8.797×10⁻⁸**, p5–p95 2.47×10⁻⁸ – 2.99×10⁻⁷, and **1.365×10⁻⁹** for EMRI-889 (the figure
  Ch 6 §4.1 prints). Recomputed by `gen_ch08.sigma_mz_measured()`, gated against F-ch06-5.
- **Treatment (worklist D5, *not* D1):** this is a both-values case, not a book-side
  correction — unlike σ_dL there is no author mandate, and the claim file really does say
  1e-4. Both values now appear, with the F-ch06-5 pointer, at all five sites the review
  found: the cold-open callout, §1's RATIFIED display equation (which carries the flag chip
  and a four-sentence both-values note), §2's needle-vs-barn-door ratio, §2's predict stem,
  and the Q8.1 area. The dossier's `M_z` row carries 889's own pair. No silent substitution
  anywhere.
- **Why the chapter survives either value:** every argument here uses only that the GW side
  is negligible against the catalogue's σ_lnM ≈ 1.28. 10⁻⁷ ≪ 1.28 for the same reason
  10⁻⁴ ≪ 1.28 is; the 193:1 one-sidedness, the sieve's 97–99% rejection and §2's vacuous
  upper leg are all unchanged. The chapter says so on the page rather than leaving the
  reader to check.
- **Author's call** (worklist §F item 2): whether `CLAIM_2D_BIAS_20260730.md:172` is amended.
  The book does not amend main-repo artifacts.

---

## Recorded decisions (no flag; logged because the worklist asks for the decision to exist)

- **D1 (σ_dL) applied.** The dossier row is now the canonical string
  `d_L  88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴`, with the canonical erratum note directly beneath
  the dossier. The erratum paragraph also states explicitly that the `M_z` row's two
  precisions are *not* an erratum but the live both-values pair of F-ch08-10 — the two
  disputes sit two rows apart on the same card and would otherwise be read as one thing.
- **D4 (spoiler discipline) applied beyond the deck.** The deck no longer prints +0.077
  (ped B1 / tomas M10); the chapter's promise is stated and the outcome is not.
  Two further pre-reveal carriers were found and fixed: (i) the **bias rail** rendered the
  `2D (mass channel) +0.077` row on page load, i.e. the answer to `#ch08-predict-1` in
  fixed chrome above the fold — it now ships with the through-Ch-7 history and arms its own
  row when the reader reaches the cold-open figure (ch03's IntersectionObserver pattern);
  (ii) **C9's binomial z = −11.86 and the 2.3–2.5× factor** were printed in §8's closing
  callout and Q8.6's answer, two chapters before their home. D4 names z = −11.86 explicitly
  as a Ch 9 reveal, so both are now the phenomenon plus `⏭ Ch 9` — the mass-aware /
  mass-blind mismatch is named, its size is not. The `<meta name="description">` was left
  alone: it carries no reveal number.
- **Traps relocated, not rewritten (ped M3).** Trap 8.A ("more information cannot hurt")
  moved from below the self-check to the end of the cold open, where the reader is actually
  thinking it; Trap 8.B ("impostor deletion is the mechanism") moved into §5, immediately
  after the paragraph that creates the misconception. Both keep their text; 8.B additionally
  gained the cell-B configuration scoping (F-ch08-9).
- **ped-m8 — the twice-stated completion trap: no ch08 change.** The worklist routes the
  "this is the one trap the book states twice, because C10 exists to retire it" annotation to
  **ch11**, not here. Ch 8's C10 trap stays exactly where it is, in §5 beside the prefactor
  paragraph it belongs to. Flagged so the ch11 agent's note has a counterpart on this side.
- **Q8.1 rewritten once, for two findings (tomas M7 + ped M2).** The old answer named the
  outcome with the *wrong* mechanism (the kinematic $M_z = M(1+z) \ge M$, which bounds the
  GW-allowed source-frame mass and says nothing about a catalogue galaxy's upper test), and
  the stem was a verbatim re-ask of `#ch08-predict-2` still framed "predict". The new item is
  in transfer form (name a population for which the upper leg is not vacuous; does the tilt
  survive?) and answers with §2's mechanism, σ_M ≳ M_g ⇒ lower edge negative ⇒ upper leg
  vacuous. The answer also ties the one-sidedness and the de-weighting to the same single
  number σ_M/M_g, and notes that §6's well-posedness defect is untouched by a better proxy —
  neither of which is answerable by scrolling up.
- **Spectral-siren sidebar added to §1 (tomas M3.5).** Placed immediately after the derivation
  box that establishes $1+z = M_z/M_g$ — the sentence that creates the misreading. It names
  Farr et al. 2019 (arXiv:1908.09084), Ezquiaga & Holz 2022 (arXiv:2202.08240) and
  Mastrogiovanni et al. 2021 (arXiv:2103.14663), and separates the two methods by *failure
  mode*, not just by definition: population-feature vs per-host association, and therefore
  proxy scatter / reference measure / d_L-derived masses (RATIFY-M7) versus an evolving mass
  function. Rendered as a visible `.callout`, not a `<details>` — a fold would miss the
  reader it exists for.
- **Rail pips converted to `Book.biasRail`'s `pips` channel** (tomas m2 / worklist §D-6).
  Integrator pass 1 shipped the capability, so the page-local `.ch08-pip` markup **and its
  CSS block are gone**; the three pips (C7, C8, RATIFY-M6 CANDIDATE) are unchanged in wording
  and tone. ch08 deliberately does **not** carry the canonical cell-B pip — D3 assigns that
  to ch07/ch09/ch10/ch11.
- **Lazy plot init (ux MAJOR-3).** All four data blocks — 8 Plotly instances — now build
  behind a `whenVisible()` `IntersectionObserver` (300 px margin), the pattern ch03 already
  uses. Browsers without `IntersectionObserver`, and print, get everything immediately.
- **`data-hypothesis` tags added to I8.2** (worklist §D-5, chapter-agent half): `#89` on the
  widget (the ratified (M1) kernel A/B — necessary, not sufficient, the standing verdict on
  the 2D pairing the dial drives) and `#71`/`#72` on the "Has anyone tried to fix this?"
  control (mass_trunc: confirmed in isolation, exonerated in the pipeline). The inline-chip
  renderer itself is integrator pass 2; today the tags feed the ledger panel's
  "dead hypotheses reachable from this page's sandboxes" seed.
- **Not changed, deliberately:** F-ch08-1's pull-range disposition (the chapter still quotes
  the recomputed +2.47…+4.74 beside the readout's +3.4…+4.5 — ch11's opening-table fix
  consumes the same pair), F-ch08-2's two dials, and everything in the reviews' PRAISE
  sections, including §5's C4 refutation, which the worklist's frozen list protects.
