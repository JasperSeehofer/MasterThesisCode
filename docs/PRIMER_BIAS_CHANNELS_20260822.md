# Primer — the bias vocabulary, from one equation

*Written 2026-08-22 for the author, to close the August vocabulary gap. Everything here is a
name for a piece of ONE equation or ONE experimental design choice. Read top to bottom once
(~30 min); the glossary at the end is the lookup table afterward. The book (Ch 0–11, designed
2026-07-31) covers the mixture itself (Ch 5) and the mass channel (Ch 8) but predates all of
this — a book addendum is proposed separately.*

## 0. The one equation

Every per-event likelihood the estimator computes, for the no-BH-mass analysis, is:

    combined(e, h)  =  [ β_G_φ(h) · L_cat(e, h)  +  B_num(e, h) ]  /  D̃_φ(h)

Read it as: *"the event's host is either one of the catalogue's candidates (first term) or a
galaxy the catalogue doesn't contain (second term), normalized by the total selected
population."* Every August word names a part of this, a way of reading it out, a switchable
variant of it, or a synthetic universe it was tested in.

## 1. LEGS — the two numerator terms

- **Catalogue leg** = `β_G_φ·L_cat`: a weighted sum over the *candidate galaxies* the GLADE
  cone search returns. β_G_φ = ∫f̄·S̄_φ·w_pop dz is its global class weight.
- **Completion leg** (also "dark leg") = `B_num`: an integral over the population the
  catalogue *misses* — (1−completeness)·population·GW-likelihood.
- `D̃_φ` is the shared normalizer (selected total mass), with a catalogue half (α_G_φ) and a
  dark half (β̄_Ḡ_φ).

**S̄_φ(z; h)** — the survival factor — is the probability a source at redshift z survives
detection selection, averaged over the black-hole mass function φ. It is THE recurring object
of August: the question "which terms carry S̄_φ, and does each numerator's S̄_φ content match
its own normalizer's?" generated most of the month's findings.

## 2. CHANNELS — three ways to read the posterior out (a *readout* choice, not a model change)

- **full** channel: the posterior from `combined` as-is — what production reports.
- **matched** channel: the *dark-conditional* posterior, `B_num/β̄_Ḡ_φ` — "if we knew the host
  is not in the catalogue, what does the completion leg alone say?" This is the surgical probe:
  it isolates the completion leg's numerator/normalizer pairing.
- **pure** channel: `full − catalogue-leg` by exact subtraction — the completion leg inside
  the mixture context.

## 3. The DECOMPOSITION — the −0.108 split three ways ⚠ vocabulary collision

The B-SEL fleet's headline bias (mean_h − 0.73 = **−0.108**) was decomposed (rows #149–#150)
into three *contributions* — regrettably also called "channels" in early reports:

| contribution | size | what it is | status |
|---|---|---|---|
| **impostor drag** | **−0.079** (73%) | catalogue candidates that are NOT the true host sit at low z and drag h down | OPEN — the venue-physics front |
| **dark-fraction tilt** | +0.055 | a composition effect of the catalogue/dark mixture | measured, understood |
| **matched-channel violation** | −0.085 | the completion leg's own broken S̄_φ pairing | **RESOLVED** (the `fused` fix; O6/O7/O8) |

**Going forward: "contribution" for these three; "channel" only for §2's readouts.**

## 4. CELLS — switchable counterfactual variants of the numerators

A *cell* is a value of an instrumentation flag that changes one numerator convention, with the
default always byte-identical to production:

- `selection_in_completion_numerator`: **off** (legacy: B_num has NO S̄_φ — the ratified
  IMPLEMENTATION-CONVENTION DEFECT, because its normalizer β̄_Ḡ_φ HAS it) vs **fused** (B_num
  carries S̄_φ — the fix, now the pin for future runs; historical runs stand on off).
- `catalogue_numerator_survival`: **off** vs **phi** ("the twin": each catalogue candidate's
  kernel carries S̄_φ(z_g) — the catalogue-leg analog of the same pairing question) vs
  **phi_flat** (a constant-S̄ kill-test variant).

## 5. ARMS and VENUES — which synthetic universe the events came from

An *arm* is one registered experimental configuration; a *venue* is the synthetic universe
class behind it. The two that matter for the August story:

- **B-SEL** (seeds 900101–900112): the realistic mirror venue — real GLADE candidates, real
  estimator — where by construction **the true host is NEVER a candidate** (all candidates are
  impostors). Powerful for mechanisms; can never certify correctness of a catalogue-leg
  change (any suppression of an all-impostor leg "helps" mechanically).
- **C-SG-F** (seeds 910101–910115): the self-generated control — events drawn from the
  estimator's own model, so every deviation from zero is the estimator's own inconsistency.
  This is where the matched-channel violation was measured at 6σ and then traced.
- **b0** (25 banked seeds): the catalogued-host venue — hosts genuinely in the candidate set;
  the only venue that can adjudicate catalogue-leg correctness (the pending identity test).

## 6. The August storyline in nine facts (each with its ledger row)

1. The C-SG control fired: the matched channel is non-zero at 6.05σ (#137→#151).
2. Its own falsifier + review traced the mechanism: the **off cell's B_num omits the S̄_φ its
   normalizer carries** (#155). Label ratified: implementation-convention defect (#157).
3. O6/O7/O8: the **fused** cell fixes it — confirmed end-to-end, fleet score residual 0.41σ,
   bias leg +0.006 ± 0.011 (#158, #161, #165). Both legs of the control's verdict closed.
4. Fixing it does NOT cure the H₀ rail — the full posterior stays railed (photo-z territory).
5. The catalogue leg has the same structural pattern ("the twin"): its numerator lacks S̄_φ
   while β_G_φ integrates it. Measured: inserting it moves the headline **+0.0155 ± 0.0037** —
   real but below the pre-frozen materiality bar (#162).
6. Decomposed: ~95% of that is per-event *level* suppression, ~0 is residual h-shape (#164).
7. The derivation fight (#166–#168): my "completed pairing" (divide the level back out) was
   REFUTED — β_G_φ over the global selection sum is a measure conversion whose S̄ cancels; the
   R-rescale re-installs the B_scale defect class. **The twin as measured is the coherent
   candidate**, valid only on the fused basis — hence the fused-basis re-measurement (FC/FT)
   now running (#169).
8. NEW, live: **r_φ** — the catalogue leg's divisor uses the mass-blind Σ³ᴰ where the pairing
   wants Σ^φ: a measured, un-derived, h-sloped ×0.886 factor on the dominant leg
   ([P3-RPHI]; rescore in flight).
9. Correctness (as opposed to leverage) of any catalogue-leg change awaits the **b0 identity
   test** — the registered next decisive measurement.

## 7. The governance words (so the reports read cleanly)

**A-numbers** (A17…A22) are process amendments — rules earned by specific failures (e.g. A20 =
every verdict gets a clean-context adversarial review before banking; A21 = the executed arm
must match its registration text; A22 = no tree changes during a registered run). **Bands** are
pre-frozen numeric verdict regions; **gates** are validity checks that VOID a run rather than
shade it; **[MEASURED]/[BANKED]/[AGENT]/[DOC]** tag how much you may lean on a number.

## 8. Glossary (lookup table)

| term | one sentence | code object / source | first row |
|---|---|---|---|
| catalogue leg | candidate-galaxy half of the mixture | `β_G_φ·L_cat` (:5399) | — |
| completion leg | missing-galaxy half | `B_num` (:5120) | — |
| S̄_φ | φ-averaged detection survival at z | `precompute_phi_marginal_survival` | #117 |
| β_G_φ / β̄_Ḡ_φ | catalogue/dark class normalizers (both S̄-weighted) | `:2065` | #118 |
| full/matched/pure | readout channels (posterior choices) | `csg_channel_scores` | #137 |
| impostor drag | −0.079 contribution: wrong-host candidates at low z | O2 read | #149 |
| dark-fraction tilt | +0.055 composition contribution | O3 read | #150 |
| matched-channel violation | −0.085 contribution: the off-cell defect | C-SG | #150/#155 |
| off / fused cell | completion numerator without/with S̄_φ | `selection_in_completion_numerator` | #117, #155 |
| the twin | the catalogue-leg S̄_φ insertion | `catalogue_numerator_survival="phi"` | #162 |
| level / shape split | per-event suppression vs residual h-tilt of the twin | shape/K-flat rescores | #164 |
| R-rescale | the refuted β_G/β_G_φ "completion" | (banked as the wrong reading) | #167–#168 |
| r_φ | Σ^φ/Σ³ᴰ ≈ 0.886, the divisor slot mismatch | `precompute_global_catalog_selection` | #168 |
| B-SEL / C-SG / b0 | venues: all-impostor / self-generated / catalogued-host | `correspondence_1d` arms | — |
| FC / FT | fused-basis coded / twin re-measurement arms | fusedarm stage | #169 |
| rail | posterior pinned at the grid edge (photo-z venue physics) | r_low | #119 |

**Where to read more:** the overnight readout (`OVERNIGHT_READOUT_20260822.md`) for the story;
`PREREGISTRATION_P3_TWIN_20260822.md` for the full P3 chain; the bias-state board (artifact)
for the visual; the ledger rows cited above for any single claim's provenance.
